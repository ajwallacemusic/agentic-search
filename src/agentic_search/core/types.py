"""Core data types shared by every module: content, filters, query ops, hits, manifests."""

from __future__ import annotations

import base64
import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Union
from urllib.parse import urlparse
from urllib.request import url2pathname

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_serializer, model_validator

# ---- Content ----------------------------------------------------------------


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"


class TextPart(BaseModel):
    kind: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    """An image, inline (`data`) or by reference (`uri`). In JSON, `data` is standard base64."""

    model_config = ConfigDict(val_json_bytes="base64")

    kind: Literal["image"] = "image"
    uri: str | None = None
    data: bytes | None = None
    mime: str = "image/png"

    @field_serializer("data", when_used="json")
    def _data_as_base64(self, data: bytes | None) -> str | None:
        return None if data is None else base64.b64encode(data).decode("ascii")

    @model_validator(mode="after")
    def _exactly_one_source(self) -> ImagePart:
        if (self.uri is None) == (self.data is None):
            raise ValueError("ImagePart needs exactly one of uri or data")
        return self


class StructuredPart(BaseModel):
    kind: Literal["structured"] = "structured"
    data: dict[str, Any]


Content = Annotated[Union[TextPart, ImagePart, StructuredPart], Field(discriminator="kind")]


def modality_of(part: Content) -> Modality:
    return Modality.IMAGE if isinstance(part, ImagePart) else Modality.TEXT


def text_of(content: list[Content]) -> str:
    """Concatenate the text-bearing parts of a content list."""
    out: list[str] = []
    for part in content:
        if isinstance(part, TextPart):
            out.append(part.text)
        elif isinstance(part, StructuredPart):
            out.append(json.dumps(part.data, default=str, sort_keys=True))
    return "\n".join(out)


def image_bytes(part: ImagePart) -> bytes:
    """Raw bytes for an inline or local-file image. Remote URIs are the caller's job."""
    if part.data is not None:
        return part.data
    assert part.uri is not None
    parsed = urlparse(part.uri)
    if parsed.scheme in ("http", "https"):
        raise ValueError("remote image URIs must be fetched by the caller")
    path = url2pathname(parsed.path) if parsed.scheme == "file" else part.uri
    return Path(path).read_bytes()


class Query(BaseModel):
    content: list[Content]

    @classmethod
    def of(cls, text: str) -> Query:
        return cls(content=[TextPart(text=text)])

    def as_text(self) -> str:
        return text_of(self.content)

    def images(self) -> list[ImagePart]:
        return [p for p in self.content if isinstance(p, ImagePart)]


class Document(BaseModel):
    doc_id: str
    content: list[Content]
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---- Filter AST -------------------------------------------------------------


class Eq(BaseModel):
    op: Literal["eq"] = "eq"
    field: str
    value: Any


class In(BaseModel):
    op: Literal["in"] = "in"
    field: str
    values: list[Any]


class Range(BaseModel):
    op: Literal["range"] = "range"
    field: str
    gte: Any = None
    gt: Any = None
    lte: Any = None
    lt: Any = None


class Exists(BaseModel):
    op: Literal["exists"] = "exists"
    field: str


class Contains(BaseModel):
    op: Literal["contains"] = "contains"
    field: str
    value: str


class And(BaseModel):
    op: Literal["and"] = "and"
    clauses: list[Filter]


class Or(BaseModel):
    op: Literal["or"] = "or"
    clauses: list[Filter]


class Not(BaseModel):
    op: Literal["not"] = "not"
    clause: Filter


Filter = Annotated[
    Union[Eq, In, Range, Exists, Contains, And, Or, Not], Field(discriminator="op")
]
for _model in (And, Or, Not):
    _model.model_rebuild()

FILTER_ADAPTER: TypeAdapter[Any] = TypeAdapter(Filter)

# ---- Query operations -------------------------------------------------------


class Capability(str, Enum):
    LEXICAL = "lexical"
    VECTOR = "vector"
    HYBRID = "hybrid"
    FILTER = "filter"
    REGEX = "regex"
    TRAVERSE = "traverse"
    AGGREGATE = "aggregate"
    FETCH = "fetch"
    NATIVE = "native"
    IMAGE_QUERY = "image_query"


class OpBase(BaseModel):
    source: str
    collection: str | None = None
    filter: Filter | None = None
    limit: int = Field(default=20, ge=1)


class Lexical(OpBase):
    type: Literal["lexical"] = "lexical"
    text: str
    fields: list[str] | None = None


class Vector(OpBase):
    type: Literal["vector"] = "vector"
    field: str
    content: Content | None = None
    hyde_text: str | None = None
    vector: list[float] | None = None  # set by the executor, never by a model

    @model_validator(mode="after")
    def _one_query(self) -> Vector:
        if (self.content is None) == (self.hyde_text is None):
            raise ValueError("vector op needs exactly one of content or hyde_text")
        return self

    def query_content(self) -> Content:
        if self.content is not None:
            return self.content
        return TextPart(text=self.hyde_text or "")


class Hybrid(OpBase):
    type: Literal["hybrid"] = "hybrid"
    text: str
    field: str
    lexical_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    vector: list[float] | None = None  # set by the executor


class FilterOnly(OpBase):
    type: Literal["filter"] = "filter"

    @model_validator(mode="after")
    def _needs_filter(self) -> FilterOnly:
        if self.filter is None:
            raise ValueError("filter_search requires a filter")
        return self


class Regex(OpBase):
    type: Literal["regex"] = "regex"
    pattern: str
    fields: list[str] | None = None


class Traverse(OpBase):
    type: Literal["traverse"] = "traverse"
    start: Filter
    rel_types: list[str] = Field(default_factory=list)
    direction: Literal["out", "in", "both"] = "both"
    depth: int = Field(default=1, ge=1, le=5)
    target_label: str | None = None


class Aggregate(OpBase):
    type: Literal["aggregate"] = "aggregate"
    group_by: list[str] = Field(min_length=1)
    metrics: list[str] = Field(default_factory=lambda: ["count"])


class Fetch(OpBase):
    type: Literal["fetch"] = "fetch"
    doc_ids: list[str] = Field(min_length=1)


class Native(OpBase):
    type: Literal["native"] = "native"
    dialect: str
    query: str


QueryOp = Annotated[
    Union[Lexical, Vector, Hybrid, FilterOnly, Regex, Traverse, Aggregate, Fetch, Native],
    Field(discriminator="type"),
]
QUERY_OP_ADAPTER: TypeAdapter[Any] = TypeAdapter(QueryOp)

_OP_CAPABILITY = {
    "lexical": Capability.LEXICAL,
    "vector": Capability.VECTOR,
    "hybrid": Capability.HYBRID,
    "filter": Capability.FILTER,
    "regex": Capability.REGEX,
    "traverse": Capability.TRAVERSE,
    "aggregate": Capability.AGGREGATE,
    "fetch": Capability.FETCH,
    "native": Capability.NATIVE,
}


def required_capabilities(op: QueryOp) -> set[Capability]:
    caps = {_OP_CAPABILITY[op.type]}
    if op.filter is not None:
        caps.add(Capability.FILTER)
    if isinstance(op, Vector) and op.content is not None:
        if modality_of(op.content) is Modality.IMAGE:
            caps.add(Capability.IMAGE_QUERY)
    return caps


# ---- Results ----------------------------------------------------------------


class OpRef(BaseModel):
    turn: int
    call_id: str
    rank: int


class Hit(BaseModel):
    doc_id: str
    source: str
    content: list[Content] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_score: float | None = None
    provenance: list[OpRef] = Field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.source}:{self.doc_id}"

    def snippet(self, chars: int = 300) -> str:
        text = text_of(self.content)
        if not text and any(isinstance(p, ImagePart) for p in self.content):
            return "[image]"
        text = " ".join(text.split())
        return text if len(text) <= chars else text[: chars - 1] + "…"


# ---- Manifests --------------------------------------------------------------


class FieldType(str, Enum):
    TEXT = "text"
    KEYWORD = "keyword"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    DATE = "date"
    VECTOR = "vector"
    IMAGE = "image"
    JSON = "json"


class FieldSpec(BaseModel):
    name: str
    type: FieldType
    searchable: bool = False
    filterable: bool = False
    sortable: bool = False
    vector_dim: int | None = None
    vector_metric: str | None = None
    embedder_id: str | None = None
    sample_values: list[Any] | None = None
    description: str | None = None

    def describe(self) -> str:
        flags = [n for n, on in (("searchable", self.searchable), ("filterable", self.filterable),
                                 ("sortable", self.sortable)) if on]
        bits = [f"`{self.name}`: {self.type.value}"]
        if flags:
            bits.append(f"[{', '.join(flags)}]")
        if self.type is FieldType.VECTOR:
            bits.append(f"(dim={self.vector_dim}, embedder={self.embedder_id})")
        if self.description:
            bits.append(f"- {self.description}")
        if self.sample_values:
            bits.append("e.g. " + ", ".join(repr(v) for v in self.sample_values[:8]))
        return " ".join(bits)


class CollectionInfo(BaseModel):
    name: str
    fields: list[FieldSpec] = Field(default_factory=list)
    count: int | None = None
    description: str | None = None

    def field(self, name: str) -> FieldSpec | None:
        return next((f for f in self.fields if f.name == name), None)


class Manifest(BaseModel):
    source: str
    backend_type: str
    capabilities: set[Capability]
    collections: list[CollectionInfo] = Field(default_factory=list)
    relationship_types: list[str] = Field(default_factory=list)
    description: str | None = None

    def resolve_collection(self, name: str | None) -> CollectionInfo | None:
        """None means 'the only collection'; returns None if ambiguous or unknown."""
        if name is None:
            return self.collections[0] if len(self.collections) == 1 else None
        return next((c for c in self.collections if c.name == name), None)

    def summary(self) -> str:
        caps = ", ".join(sorted(c.value for c in self.capabilities))
        lines = [f"## source `{self.source}` ({self.backend_type}); capabilities: {caps}"]
        if self.description:
            lines.append(self.description)
        for c in self.collections:
            count = f", {c.count} items" if c.count is not None else ""
            desc = f": {c.description}" if c.description else ""
            lines.append(f"- collection `{c.name}`{count}{desc}")
            lines.extend(f"  - {f.describe()}" for f in c.fields)
        if self.relationship_types:
            lines.append("- relationship types: " + ", ".join(self.relationship_types))
        return "\n".join(lines)


# ---- Run control ------------------------------------------------------------


class ModelUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def plus(self, other: ModelUsage) -> ModelUsage:
        return ModelUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


class Budget(BaseModel):
    max_turns: int = 4
    max_tool_calls: int = 32
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_seconds: float | None = 60.0


class StopReason(str, Enum):
    CONTROLLER_STOP = "controller_stop"
    NO_PLAN = "no_plan"
    SINGLE_PASS = "single_pass"
    DELEGATE_DONE = "delegate_done"
    BUDGET_TURNS = "budget_turns"
    BUDGET_TOOL_CALLS = "budget_tool_calls"
    BUDGET_TOKENS = "budget_tokens"
    BUDGET_COST = "budget_cost"
    BUDGET_TIME = "budget_time"


class ToolError(BaseModel):
    call_id: str
    kind: Literal["validation", "backend", "timeout", "embedder", "policy"]
    message: str
    source: str | None = None

    def render(self) -> str:
        return f"ERROR [{self.kind}] {self.message}"
