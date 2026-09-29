# Plan 1: Core Harness + End-to-End on Local Data — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working `agentic_search` Python library covering the core types, the Plan→Execute→Analyze→Decide loop with swappable roles, the three modes, a plain-files backend, local embedders, Anthropic and OpenAI-compatible drivers, LLM and cross-encoder judges, YAML config, and a BEIR eval that compares modes on NFCorpus.

**Architecture:** Every datastore sits behind the `Backend` protocol (`discover` → `Manifest`, `execute(QueryOp)` → `list[Hit]`). Models fill two roles: a `Driver` plans and calls tools, and a `Decider` judges relevance and/or decides whether to stop. Each `Harness.search()` builds an Executor, Analyzer, Controller and Planner, then runs the loop over a shared `SearchState`.

**Tech Stack:** Python 3.12 (uv), pydantic v2, asyncio, numpy, rank-bm25, PyYAML, pytest + pytest-asyncio. Optional: anthropic, openai, sentence-transformers.

**Spec:** `docs/superpowers/specs/2026-09-29-agentic-search-harness-design.md`

**Follow-on plans:** Plan 2 covers the SQL/search backends (pgvector, MySQL, BigQuery, OpenSearch), the native-query guard, and the docker-compose contract suite. Plan 3 covers Neo4j, Milvus, the remote embedders (Vertex, TEI, generic HTTP, GCP/Azure auth), and TypeSafe deciders. Both build on the interfaces this plan defines.

## Global Constraints

- Python `>=3.11`, pydantic v2, async throughout.
- Read-only: no adapter ever issues writes.
- Tool and validation failures become `ToolError`s returned to the planner and are never raised. Only harness-level failures raise `HarnessError`.
- Every payload bound for an external model or embedder passes through `Hooks.before_model_call(model_id, payload)`.
- A vector op never falls back to a different embedder than the one bound to the field (`FieldSpec.embedder_id`).
- Secrets come from the environment only (`*_env` keys in config). Never write them to traces.
- Run tests with `uv run pytest`. Tests make no network calls unless marked `live`.
- Commit after every task. Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## File Map

```
pyproject.toml, .python-version, .gitignore, README.md
src/agentic_search/
  __init__.py            exports Harness, SearchResult, Query, Budget
  testing.py             deterministic fakes (drivers, deciders, LLM client)
  config.py              YAML/dict → Harness; type registries
  core/types.py          content, filter AST, query ops, hits, manifests, budget, errors
  core/state.py          CandidatePool, Trace, Usage, SearchState
  core/hooks.py          Hooks, SourcePolicy
  core/annotations.py    apply_annotations(manifest, ann)
  core/harness.py        Harness, SearchResult, modes, delegate runtime
  backends/base.py       Backend protocol, BackendError, rrf_merge
  backends/filters.py    matches(), filter_fields()
  backends/files.py      FilesBackend
  embedders/base.py      Embedder protocol, registry, CachedEmbedder, vector math
  embedders/local.py     HashEmbedder, SentenceTransformerEmbedder
  models/base.py         Driver/Decider protocols and their data types
  models/llm.py          LLMClient protocol, ChatMessage, ChatResponse
  models/anthropic.py    AnthropicClient
  models/openai_compat.py OpenAICompatClient
  models/driver.py       ToolCallingDriver
  models/llm_judge.py    LLMJudge
  models/cross_encoder.py CrossEncoderJudge
  roles/tools.py         build_tool_specs, parse_call
  roles/executor.py      Executor
  roles/analyzer.py      Analyzer
  roles/controller.py    Controller
  roles/planner.py       Planner, render_manifests
  eval/datasets.py  eval/metrics.py  eval/runner.py
scripts/eval_beir.py
tests/…                  one test module per source module
```

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`
- Create: `src/agentic_search/__init__.py`, plus empty `__init__.py` in `core/`, `backends/`, `embedders/`, `models/`, `roles/` and `eval/`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Produces: the importable package `agentic_search` with `__version__ = "0.1.0"`.

- [ ] **Step 1: Write config files**

`pyproject.toml`:
```toml
[project]
name = "agentic-search"
version = "0.1.0"
description = "Agentic search harness: plan, execute, analyze, decide over any datastore with swappable models."
requires-python = ">=3.11"
dependencies = [
  "pydantic>=2.7",
  "numpy>=1.26",
  "rank-bm25>=0.2.2",
  "pyyaml>=6.0",
]

[project.optional-dependencies]
anthropic = ["anthropic>=0.40"]
openai = ["openai>=1.50"]
local = ["sentence-transformers>=3.0", "pillow>=10"]

[dependency-groups]
dev = [
  "pytest>=8",
  "pytest-asyncio>=0.24",
  "ruff>=0.6",
  "anthropic>=0.40",
  "openai>=1.50",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/agentic_search"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
  "live: calls real network services (skipped by default)",
  "slow: downloads or runs local models",
]
addopts = "-m 'not live and not slow'"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "I"]
```

`.python-version`:
```
3.12
```

`.gitignore`:
```
.venv/
__pycache__/
.pytest_cache/
.ruff_cache/
data/
eval-results/
*.egg-info/
```

`src/agentic_search/__init__.py`:
```python
"""Agentic search harness."""

__version__ = "0.1.0"
```

Create empty files: `src/agentic_search/{core,backends,embedders,models,roles,eval}/__init__.py`.

- [ ] **Step 2: Write the smoke test**

`tests/test_smoke.py`:
```python
import agentic_search


def test_version():
    assert agentic_search.__version__ == "0.1.0"
```

- [ ] **Step 3: Install and run**

Run: `uv sync && uv run pytest -q`
Expected: `1 passed`

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: scaffold agentic-search package

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Core types

**Files:**
- Create: `src/agentic_search/core/types.py`
- Test: `tests/core/test_types.py` (plus an empty `tests/core/__init__.py`, so test module names don't collide)

**Interfaces:**
- Produces:
  - Content: `Modality`, `TextPart`, `ImagePart`, `StructuredPart`, `Content`, `modality_of()`, `text_of()`, `image_bytes()`, `Query`, `Document`
  - Filter AST: `Eq`, `In`, `Range`, `Exists`, `Contains`, `And`, `Or`, `Not`, `Filter`, `FILTER_ADAPTER`
  - Query ops: `Capability`, `OpBase`, `Lexical`, `Vector`, `Hybrid`, `FilterOnly`, `Regex`, `Traverse`, `Aggregate`, `Fetch`, `Native`, `QueryOp`, `QUERY_OP_ADAPTER`, `required_capabilities()`
  - Results and schema: `OpRef`, `Hit`, `FieldType`, `FieldSpec`, `CollectionInfo`, `Manifest`
  - Run control: `ModelUsage`, `Budget`, `StopReason`, `ToolError`

- [ ] **Step 1: Write the failing tests**

`tests/core/test_types.py`:
```python
import pytest
from pydantic import ValidationError

from agentic_search.core.types import (
    FILTER_ADAPTER, QUERY_OP_ADAPTER, And, Budget, Capability, CollectionInfo, Eq, FieldSpec,
    FieldType, FilterOnly, Hit, ImagePart, Lexical, Manifest, ModelUsage, Not, Query, TextPart,
    ToolError, Vector, image_bytes, required_capabilities, text_of, StructuredPart,
)


def test_query_of_and_text():
    q = Query.of("hello")
    assert q.as_text() == "hello"
    assert q.images() == []


def test_text_of_includes_structured():
    assert text_of([TextPart(text="a"), StructuredPart(data={"b": 1})]) == 'a\n{"b": 1}'


def test_image_part_requires_exactly_one_source():
    with pytest.raises(ValidationError):
        ImagePart()
    with pytest.raises(ValidationError):
        ImagePart(uri="file:///x.png", data=b"x")


def test_image_bytes_from_data_and_file(tmp_path):
    assert image_bytes(ImagePart(data=b"abc")) == b"abc"
    p = tmp_path / "x.png"
    p.write_bytes(b"png")
    assert image_bytes(ImagePart(uri=p.as_uri())) == b"png"
    with pytest.raises(ValueError):
        image_bytes(ImagePart(uri="https://example.com/x.png"))


def test_filter_roundtrip_nested():
    f = FILTER_ADAPTER.validate_python({
        "op": "and",
        "clauses": [
            {"op": "eq", "field": "a", "value": 1},
            {"op": "not", "clause": {"op": "exists", "field": "b"}},
        ],
    })
    assert isinstance(f, And)
    assert isinstance(f.clauses[1], Not)


def test_query_op_discriminates_and_defaults():
    op = QUERY_OP_ADAPTER.validate_python({"type": "lexical", "source": "s", "text": "x"})
    assert isinstance(op, Lexical)
    assert op.limit == 20


def test_vector_needs_exactly_one_query():
    with pytest.raises(ValidationError):
        Vector(source="s", field="e")
    v = Vector(source="s", field="e", hyde_text="hyp")
    assert v.query_content() == TextPart(text="hyp")


def test_filter_only_requires_filter():
    with pytest.raises(ValidationError):
        FilterOnly(source="s")


def test_required_capabilities():
    op = Vector(source="s", field="e", content=ImagePart(uri="file:///a.png"),
                filter=Eq(field="x", value=1))
    assert required_capabilities(op) == {
        Capability.VECTOR, Capability.FILTER, Capability.IMAGE_QUERY,
    }


def test_hit_key_and_snippet():
    h = Hit(doc_id="d", source="s", content=[TextPart(text="a  b\n" + "c" * 50)])
    assert h.key == "s:d"
    snip = h.snippet(10)
    assert snip.startswith("a b c") and len(snip) == 10
    assert Hit(doc_id="i", source="s", content=[ImagePart(data=b"x")]).snippet() == "[image]"


def test_manifest_resolve_and_summary():
    one = CollectionInfo(name="notes", fields=[
        FieldSpec(name="body", type=FieldType.TEXT, searchable=True),
        FieldSpec(name="emb", type=FieldType.VECTOR, vector_dim=8, embedder_id="hash"),
    ], count=3)
    m = Manifest(source="src", backend_type="files",
                 capabilities={Capability.LEXICAL}, collections=[one])
    assert m.resolve_collection(None) is one
    assert m.resolve_collection("nope") is None
    two = m.model_copy(update={"collections": [one, CollectionInfo(name="other")]})
    assert two.resolve_collection(None) is None
    s = m.summary()
    assert "`src`" in s and "`notes`" in s and "embedder=hash" in s and "3 items" in s


def test_model_usage_plus_and_budget_defaults():
    u = ModelUsage(input_tokens=1, output_tokens=2, cost_usd=0.5).plus(
        ModelUsage(input_tokens=3, output_tokens=4, cost_usd=0.25))
    assert (u.input_tokens, u.output_tokens, u.cost_usd) == (4, 6, 0.75)
    assert Budget().max_turns == 4


def test_tool_error_render():
    e = ToolError(call_id="c1", kind="validation", message="bad field")
    assert e.render() == "ERROR [validation] bad field"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/core/test_types.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_search.core.types'`

- [ ] **Step 3: Implement**

`src/agentic_search/core/types.py`:
```python
"""Core data types shared by every module: content, filters, query ops, hits, manifests."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Union
from urllib.parse import urlparse
from urllib.request import url2pathname

from pydantic import BaseModel, Field, TypeAdapter, model_validator

# ---- Content ----------------------------------------------------------------


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"


class TextPart(BaseModel):
    kind: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    kind: Literal["image"] = "image"
    uri: str | None = None
    data: bytes | None = None
    mime: str = "image/png"

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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/core/test_types.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(core): content, filter AST, query ops, hits, manifests

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Filter evaluation

**Files:**
- Create: `src/agentic_search/backends/filters.py`
- Test: `tests/backends/test_filters.py` (plus an empty `tests/backends/__init__.py`)

**Interfaces:**
- Consumes: the filter AST from Task 2.
- Produces: `matches(f: Filter, record: dict) -> bool` and `filter_fields(f: Filter) -> set[str]`. Plans 2 and 3 add dialect translators to this module.

- [ ] **Step 1: Write the failing tests**

`tests/backends/test_filters.py`:
```python
from agentic_search.backends.filters import filter_fields, matches
from agentic_search.core.types import And, Contains, Eq, Exists, In, Not, Or, Range

REC = {"type": "drug", "year": 2020, "tags": ["a", "b"], "date": "2024-03-01",
       "nested": {"x": 5}, "none": None}


def test_eq_scalar_and_list():
    assert matches(Eq(field="type", value="drug"), REC)
    assert not matches(Eq(field="type", value="history"), REC)
    assert matches(Eq(field="tags", value="a"), REC)


def test_in():
    assert matches(In(field="year", values=[2019, 2020]), REC)
    assert matches(In(field="tags", values=["z", "b"]), REC)
    assert not matches(In(field="year", values=[1]), REC)


def test_range_numbers_and_iso_dates():
    assert matches(Range(field="year", gte=2020, lt=2021), REC)
    assert not matches(Range(field="year", gt=2020), REC)
    assert matches(Range(field="date", gte="2024-01-01"), REC)


def test_range_type_mismatch_is_false():
    assert not matches(Range(field="type", gte=3), REC)


def test_exists_and_missing():
    assert matches(Exists(field="type"), REC)
    assert not matches(Exists(field="none"), REC)
    assert not matches(Exists(field="missing"), REC)
    assert not matches(Eq(field="missing", value=1), REC)


def test_contains_case_insensitive():
    assert matches(Contains(field="type", value="DR"), REC)
    assert matches(Contains(field="tags", value="b"), REC)


def test_dotted_path():
    assert matches(Eq(field="nested.x", value=5), REC)


def test_boolean_combinators():
    f = And(clauses=[Eq(field="type", value="drug"),
                     Or(clauses=[Eq(field="year", value=1), Not(clause=Eq(field="year", value=2))])])
    assert matches(f, REC)


def test_filter_fields():
    f = And(clauses=[Eq(field="a", value=1), Not(clause=Or(clauses=[Exists(field="b")]))])
    assert filter_fields(f) == {"a", "b"}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/backends/test_filters.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`src/agentic_search/backends/filters.py`:
```python
"""Filter AST utilities. Plans 2 and 3 add translators to backend dialects here."""

from __future__ import annotations

import operator
from typing import Any, Callable

from agentic_search.core.types import And, Contains, Eq, Exists, Filter, In, Not, Or, Range

_MISSING = object()


def _lookup(record: dict[str, Any], field: str) -> Any:
    if field in record:
        return record[field]
    cur: Any = record
    for part in field.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return _MISSING
    return cur


def _compare(value: Any, bound: Any, fn: Callable[[Any, Any], bool]) -> bool:
    try:
        return bool(fn(value, bound))
    except TypeError:
        return False


def matches(f: Filter, record: dict[str, Any]) -> bool:
    """Evaluate a filter against a flat or nested dict (used by in-memory backends)."""
    if isinstance(f, And):
        return all(matches(c, record) for c in f.clauses)
    if isinstance(f, Or):
        return any(matches(c, record) for c in f.clauses)
    if isinstance(f, Not):
        return not matches(f.clause, record)
    value = _lookup(record, f.field)
    if isinstance(f, Exists):
        return value is not _MISSING and value is not None
    if value is _MISSING or value is None:
        return False
    if isinstance(f, Eq):
        return f.value in value if isinstance(value, list) else value == f.value
    if isinstance(f, In):
        if isinstance(value, list):
            return any(v in f.values for v in value)
        return value in f.values
    if isinstance(f, Contains):
        if isinstance(value, list):
            return f.value in value
        return f.value.lower() in str(value).lower()
    if isinstance(f, Range):
        checks = ((f.gte, operator.ge), (f.gt, operator.gt), (f.lte, operator.le), (f.lt, operator.lt))
        return all(_compare(value, bound, fn) for bound, fn in checks if bound is not None)
    raise TypeError(f"unknown filter node {type(f).__name__}")


def filter_fields(f: Filter) -> set[str]:
    """Every field name a filter references."""
    if isinstance(f, (And, Or)):
        return set().union(*(filter_fields(c) for c in f.clauses))
    if isinstance(f, Not):
        return filter_fields(f.clause)
    return {f.field}
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/backends/test_filters.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(backends): in-memory filter evaluation

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---
### Task 4: Embedders (protocol, registry, cache, local)

**Files:**
- Create: `src/agentic_search/embedders/base.py`, `src/agentic_search/embedders/local.py`
- Test: `tests/embedders/test_embedders.py` (plus an empty `tests/embedders/__init__.py`)

**Interfaces:**
- Consumes: `Content`, `Modality`, `TextPart`, `ImagePart`, `modality_of`, `text_of`, `image_bytes` (Task 2).
- Produces:
  - Protocol and types: `Embedding = list[float]`, `Purpose = Literal["query","document"]`, and the `Embedder` protocol (`id: str`, `modalities: set[Modality]`, `dim` (property), `async embed(items: list[Content], purpose: Purpose) -> list[Embedding]`).
  - Helpers: `supports(embedder, part) -> bool`, `normalize_rows(np.ndarray) -> np.ndarray`, `cosine_scores(query, matrix) -> np.ndarray`.
  - `EmbedderRegistry(embedders=())` with `.register(e)`, `.get(id) -> Embedder | None`, `in` and `.ids()`.
  - `CachedEmbedder(inner, maxsize=1024)`.
  - `HashEmbedder(dim=256, id="hash")` and `SentenceTransformerEmbedder(model_name, *, id=None, image=False, query_prefix="", document_prefix="", device=None)`.

- [ ] **Step 1: Write the failing tests**

`tests/embedders/test_embedders.py`:
```python
import numpy as np
import pytest

from agentic_search.core.types import ImagePart, Modality, TextPart
from agentic_search.embedders.base import (
    CachedEmbedder, EmbedderRegistry, cosine_scores, normalize_rows, supports,
)
from agentic_search.embedders.local import HashEmbedder


async def test_hash_embedder_similarity_and_determinism():
    e = HashEmbedder(dim=128)
    a, b, c = await e.embed(
        [TextPart(text="cat sat on mat"), TextPart(text="the cat on a mat"),
         TextPart(text="quarterly revenue report")], "document")
    assert len(a) == 128
    assert np.dot(a, b) > np.dot(a, c)
    assert (await e.embed([TextPart(text="cat sat on mat")], "query"))[0] == a


async def test_hash_embedder_rejects_images():
    with pytest.raises(ValueError):
        await HashEmbedder().embed([ImagePart(data=b"x")], "query")


def test_supports():
    e = HashEmbedder()
    assert supports(e, TextPart(text="x"))
    assert not supports(e, ImagePart(data=b"x"))


def test_cosine_scores():
    m = normalize_rows(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32))
    s = cosine_scores([3.0, 0.0], m)
    assert s[0] == pytest.approx(1.0) and s[1] == pytest.approx(0.0)
    assert cosine_scores([0.0, 0.0], m).tolist() == [0.0, 0.0]


def test_registry():
    e = HashEmbedder(id="h1")
    reg = EmbedderRegistry([e])
    assert "h1" in reg and reg.get("h1") is e and reg.get("nope") is None
    reg.register(e)  # re-registering the same object is fine
    with pytest.raises(ValueError):
        reg.register(HashEmbedder(id="h1"))
    assert reg.ids() == ["h1"]


class CountingEmbedder:
    id = "count"
    modalities = {Modality.TEXT}
    dim = 2

    def __init__(self):
        self.calls = 0

    async def embed(self, items, purpose):
        self.calls += 1
        return [[float(len(i.text)), 1.0] for i in items]


async def test_cached_embedder_caches_query_text():
    inner = CountingEmbedder()
    c = CachedEmbedder(inner, maxsize=2)
    v1 = await c.embed([TextPart(text="ab")], "query")
    v2 = await c.embed([TextPart(text="ab")], "query")
    assert v1 == v2 == [[2.0, 1.0]] and inner.calls == 1
    await c.embed([TextPart(text="ab")], "document")  # documents bypass the cache
    assert inner.calls == 2
    assert c.id == "count" and c.dim == 2


@pytest.mark.slow
async def test_sentence_transformer_embedder():
    pytest.importorskip("sentence_transformers")
    from agentic_search.embedders.local import SentenceTransformerEmbedder
    e = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    [v] = await e.embed([TextPart(text="hello")], "query")
    assert len(v) == e.dim == 384
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/embedders -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement base**

`src/agentic_search/embedders/base.py`:
```python
"""Embedder protocol, registry, query cache, and vector math helpers."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Literal, Protocol, runtime_checkable

import numpy as np

from agentic_search.core.types import Content, Modality, TextPart, modality_of

Embedding = list[float]
Purpose = Literal["query", "document"]


@runtime_checkable
class Embedder(Protocol):
    id: str
    modalities: set[Modality]

    @property
    def dim(self) -> int: ...

    async def embed(self, items: list[Content], purpose: Purpose) -> list[Embedding]: ...


def supports(embedder: Embedder, part: Content) -> bool:
    return modality_of(part) in embedder.modalities


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine_scores(query: Embedding, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of `query` against each row. Rows must already be L2-normalized."""
    q = np.asarray(query, dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n == 0.0:
        return np.zeros(len(matrix), dtype=np.float32)
    return matrix @ (q / n)


class EmbedderRegistry:
    """Maps embedder ids (as recorded on manifest vector fields) to embedders."""

    def __init__(self, embedders: Iterable[Embedder] = ()):
        self._by_id: dict[str, Embedder] = {}
        for e in embedders:
            self.register(e)

    def register(self, embedder: Embedder) -> None:
        existing = self._by_id.get(embedder.id)
        if existing is not None and existing is not embedder:
            raise ValueError(f"duplicate embedder id {embedder.id!r}")
        self._by_id[embedder.id] = embedder

    def get(self, embedder_id: str) -> Embedder | None:
        return self._by_id.get(embedder_id)

    def __contains__(self, embedder_id: object) -> bool:
        return embedder_id in self._by_id

    def ids(self) -> list[str]:
        return sorted(self._by_id)


class CachedEmbedder:
    """LRU cache for text query embeddings; planners often re-embed similar rewrites."""

    def __init__(self, inner: Embedder, maxsize: int = 1024):
        self.inner = inner
        self.id = inner.id
        self.modalities = inner.modalities
        self.maxsize = maxsize
        self._cache: OrderedDict[str, Embedding] = OrderedDict()

    @property
    def dim(self) -> int:
        return self.inner.dim

    async def embed(self, items: list[Content], purpose: Purpose) -> list[Embedding]:
        if purpose != "query" or not all(isinstance(i, TextPart) for i in items):
            return await self.inner.embed(items, purpose)
        texts = [i.text for i in items]  # type: ignore[union-attr]
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if missing:
            vectors = await self.inner.embed([TextPart(text=t) for t in missing], "query")
            self._cache.update(zip(missing, vectors))
        out = []
        for t in texts:
            self._cache.move_to_end(t)
            out.append(self._cache[t])
        while len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
        return out
```

- [ ] **Step 4: Implement local embedders**

`src/agentic_search/embedders/local.py`:
```python
"""Local embedders: a zero-dependency hash embedder and sentence-transformers (text or CLIP)."""

from __future__ import annotations

import asyncio
import hashlib
import io
import re
from typing import Any

import numpy as np

from agentic_search.core.types import Content, ImagePart, Modality, image_bytes, text_of
from agentic_search.embedders.base import Embedding, Purpose

_TOKEN = re.compile(r"\w+")


class HashEmbedder:
    """Deterministic bag-of-words feature hashing. No downloads; for tests, demos and CI."""

    def __init__(self, dim: int = 256, id: str = "hash"):
        self.id = id
        self.dim = dim
        self.modalities = {Modality.TEXT}

    def _vector(self, text: str) -> Embedding:
        v = np.zeros(self.dim, dtype=np.float32)
        for token in _TOKEN.findall(text.lower()):
            v[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = float(np.linalg.norm(v))
        return (v / n if n else v).tolist()

    async def embed(self, items: list[Content], purpose: Purpose = "document") -> list[Embedding]:
        out = []
        for item in items:
            if isinstance(item, ImagePart):
                raise ValueError(f"{self.id} cannot embed images")
            out.append(self._vector(text_of([item])))
        return out


class SentenceTransformerEmbedder:
    """sentence-transformers model; set image=True for CLIP-style models. Needs the `local` extra."""

    def __init__(self, model_name: str, *, id: str | None = None, image: bool = False,
                 query_prefix: str = "", document_prefix: str = "", device: str | None = None):
        self.model_name = model_name
        self.id = id or f"st:{model_name}"
        self.modalities = {Modality.TEXT, Modality.IMAGE} if image else {Modality.TEXT}
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.device = device
        self._model: Any = None
        self._dim: int | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            d = self.model.get_sentence_embedding_dimension()
            self._dim = int(d) if d else len(self.model.encode(["x"])[0])
        return self._dim

    async def embed(self, items: list[Content], purpose: Purpose) -> list[Embedding]:
        return await asyncio.to_thread(self._encode, items, purpose)

    def _encode(self, items: list[Content], purpose: Purpose) -> list[Embedding]:
        prefix = self.query_prefix if purpose == "query" else self.document_prefix
        inputs: list[Any] = []
        for item in items:
            if isinstance(item, ImagePart):
                if Modality.IMAGE not in self.modalities:
                    raise ValueError(f"{self.id} is text-only")
                from PIL import Image
                inputs.append(Image.open(io.BytesIO(image_bytes(item))).convert("RGB"))
            else:
                inputs.append(prefix + text_of([item]))
        vectors = self.model.encode(inputs, normalize_embeddings=True, convert_to_numpy=True)
        return [v.tolist() for v in vectors]
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/embedders -q`
Expected: all pass (the slow test is deselected)

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(embedders): protocol, registry, query cache, hash and sentence-transformers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Backend protocol + FilesBackend

**Files:**
- Create: `src/agentic_search/backends/base.py`, `src/agentic_search/backends/files.py`
- Create: `tests/conftest.py` (shared fixtures used by later tasks)
- Test: `tests/backends/test_files.py`

**Interfaces:**
- Consumes: Tasks 2–4.
- Produces:
  - `DiscoverDetail = Literal["summary","full"]`, `BackendError`, `UnsupportedOperation(BackendError)`.
  - The `Backend` protocol: `name: str`, `backend_type: str`, `capabilities() -> set[Capability]`, `async discover(detail="full", collection=None) -> Manifest`, `async execute(op) -> list[Hit]`, `async close()`.
  - `rrf_merge(rankings: list[tuple[list[str], float]], k=60) -> list[tuple[str, float]]`.
  - `FilesBackend(name, root=None, *, documents=None, embedder=None, glob="**/*", collection="files", description=None, max_file_bytes=2_000_000)` and `FilesBackend.from_documents(name, documents, **kw)`.
  - The FilesBackend manifest has one collection named `collection`. Its fields are `text` (TEXT, searchable), inferred metadata fields (filterable), and `embedding` (VECTOR, `embedder_id=embedder.id`) when an embedder is set. Doc ids are the relative path, or `Document.doc_id` for in-memory documents.
  - `infer_field_type(values) -> FieldType`.
  - Fixtures: `medical_docs`, which returns a `list[Document]` with ids d1–d5, and `docs_backend`, a `FilesBackend` named `"docs"` with a `HashEmbedder()`.

- [ ] **Step 1: Write shared fixtures**

`tests/conftest.py`:
```python
import pytest

from agentic_search.backends.files import FilesBackend
from agentic_search.core.types import Document, TextPart
from agentic_search.embedders.local import HashEmbedder

_ROWS = [
    ("d1", "Aspirin reduces fever and relieves headache pain", {"type": "drug", "year": 2020}),
    ("d2", "Ibuprofen is an anti-inflammatory used for pain", {"type": "drug", "year": 2021}),
    ("d3", "The history of the printing press in Europe", {"type": "history", "year": 1999}),
    ("d4", "Acetaminophen treats headache and fever", {"type": "drug", "year": 2019}),
    ("d5", "Medieval castles and their architecture", {"type": "history", "year": 2005}),
]


@pytest.fixture
def medical_docs() -> list[Document]:
    return [Document(doc_id=i, content=[TextPart(text=t)], metadata=m) for i, t, m in _ROWS]


@pytest.fixture
def docs_backend(medical_docs) -> FilesBackend:
    return FilesBackend.from_documents("docs", medical_docs, embedder=HashEmbedder())
```

- [ ] **Step 2: Write the failing tests**

`tests/backends/test_files.py`:
```python
import pytest

from agentic_search.backends.base import Backend, BackendError, UnsupportedOperation, rrf_merge
from agentic_search.backends.files import FilesBackend, infer_field_type
from agentic_search.core.types import (
    Aggregate, Capability, Eq, Fetch, FieldType, FilterOnly, Hybrid, Lexical, Regex, StructuredPart,
    TextPart, Traverse, Vector,
)
from agentic_search.embedders.local import HashEmbedder


@pytest.fixture
def folder(tmp_path):
    (tmp_path / "a.md").write_text("The cat sat on the mat")
    (tmp_path / "b.txt").write_text("Dogs chase cats in the park")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.md").write_text("Quarterly revenue grew 10 percent")
    (tmp_path / "img.png").write_bytes(b"\x89PNG fake")
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01")
    return tmp_path


def test_rrf_merge():
    fused = rrf_merge([(["a", "b"], 1.0), (["b", "c"], 1.0)], k=0)
    assert fused[0][0] == "b"


def test_infer_field_type():
    assert infer_field_type([1, 2]) is FieldType.INT
    assert infer_field_type([1, 2.5]) is FieldType.FLOAT
    assert infer_field_type([True]) is FieldType.BOOL
    assert infer_field_type(["2024-01-01T00:00:00+00:00"]) is FieldType.DATE
    assert infer_field_type(["x"]) is FieldType.KEYWORD
    assert infer_field_type([{"a": 1}]) is FieldType.JSON


async def test_discover_folder(folder):
    b = FilesBackend("fs", folder)
    assert isinstance(b, Backend)
    m = await b.discover()
    coll = m.resolve_collection(None)
    assert coll.name == "files" and coll.count == 4
    names = {f.name for f in coll.fields}
    assert {"text", "path", "ext", "size", "modified"} <= names
    assert coll.field("size").type is FieldType.INT
    assert coll.field("modified").type is FieldType.DATE
    assert set(coll.field("ext").sample_values) == {".md", ".txt", ".png"}
    assert Capability.VECTOR not in m.capabilities


async def test_lexical_exact_token(folder):
    b = FilesBackend("fs", folder)
    hits = await b.execute(Lexical(source="fs", text="cat"))
    assert [h.doc_id for h in hits] == ["a.md"]
    assert hits[0].raw_score > 0 and hits[0].metadata["ext"] == ".md"


async def test_filter_regex_fetch_aggregate(folder):
    b = FilesBackend("fs", folder)
    md = await b.execute(FilterOnly(source="fs", filter=Eq(field="ext", value=".md")))
    assert {h.doc_id for h in md} == {"a.md", "sub/c.md"}
    rx = await b.execute(Regex(source="fs", pattern=r"\d+ percent"))
    assert [h.doc_id for h in rx] == ["sub/c.md"]
    f = await b.execute(Fetch(source="fs", doc_ids=["b.txt", "missing"]))
    assert [h.doc_id for h in f] == ["b.txt"]
    agg = await b.execute(Aggregate(source="fs", group_by=["ext"]))
    top = agg[0].content[0]
    assert isinstance(top, StructuredPart) and top.data == {"ext": ".md", "count": 2}


async def test_bad_regex_and_unsupported(folder):
    b = FilesBackend("fs", folder)
    with pytest.raises(BackendError):
        await b.execute(Regex(source="fs", pattern="("))
    with pytest.raises(UnsupportedOperation):
        await b.execute(Traverse(source="fs", start=Eq(field="x", value=1)))
    with pytest.raises(UnsupportedOperation):
        await b.execute(Aggregate(source="fs", group_by=["ext"], metrics=["sum"]))


async def test_documents_vector_and_hybrid(docs_backend):
    m = await docs_backend.discover()
    coll = m.resolve_collection(None)
    assert coll.field("embedding").embedder_id == "hash"
    assert coll.field("year").type is FieldType.INT
    assert {Capability.VECTOR, Capability.HYBRID} <= m.capabilities
    [qv] = await HashEmbedder().embed([TextPart(text="headache fever")], "query")
    vec = await docs_backend.execute(Vector(source="docs", field="embedding", hyde_text="x", vector=qv))
    assert vec[0].doc_id in {"d1", "d4"}
    hyb = await docs_backend.execute(
        Hybrid(source="docs", field="embedding", text="headache", vector=qv, limit=2))
    assert {h.doc_id for h in hyb} == {"d1", "d4"}


async def test_lexical_with_filter(docs_backend):
    hits = await docs_backend.execute(
        Lexical(source="docs", text="pain", filter=Eq(field="year", value=2021)))
    assert [h.doc_id for h in hits] == ["d2"]


async def test_vector_without_resolved_vector_fails(docs_backend):
    with pytest.raises(BackendError):
        await docs_backend.execute(Vector(source="docs", field="embedding", hyde_text="x"))


async def test_unknown_collection(docs_backend):
    with pytest.raises(BackendError):
        await docs_backend.execute(Lexical(source="docs", collection="nope", text="x"))


def test_requires_exactly_one_source(medical_docs, tmp_path):
    with pytest.raises(ValueError):
        FilesBackend("x")
    with pytest.raises(ValueError):
        FilesBackend("x", tmp_path, documents=medical_docs)
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/backends/test_files.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement the backend protocol**

`src/agentic_search/backends/base.py`:
```python
"""Backend protocol: every datastore adapter implements this."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from agentic_search.core.types import Capability, Hit, Manifest, QueryOp

DiscoverDetail = Literal["summary", "full"]


class BackendError(Exception):
    """Raised by adapters for query failures; the executor turns it into a ToolError."""


class UnsupportedOperation(BackendError):
    """The adapter does not implement this op (or this variant of it)."""


@runtime_checkable
class Backend(Protocol):
    name: str
    backend_type: str

    def capabilities(self) -> set[Capability]: ...

    async def discover(self, detail: DiscoverDetail = "full",
                       collection: str | None = None) -> Manifest: ...

    async def execute(self, op: QueryOp) -> list[Hit]: ...

    async def close(self) -> None: ...


def rrf_merge(rankings: list[tuple[list[str], float]], k: int = 60) -> list[tuple[str, float]]:
    """Weighted reciprocal rank fusion. rankings = [(ids_best_first, weight), ...]."""
    scores: dict[str, float] = {}
    for ids, weight in rankings:
        for rank, doc_id in enumerate(ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
```

- [ ] **Step 5: Implement FilesBackend**

`src/agentic_search/backends/files.py`:
```python
"""Plain-files backend: a directory (or in-memory documents) searched with BM25, regex,
metadata filters, and an optional local vector index."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from rank_bm25 import BM25Okapi

from agentic_search.backends.base import (
    BackendError, DiscoverDetail, UnsupportedOperation, rrf_merge,
)
from agentic_search.backends.filters import matches
from agentic_search.core.types import (
    Aggregate, Capability, CollectionInfo, Content, Document, Fetch, FieldSpec, FieldType, Filter,
    FilterOnly, Hit, Hybrid, ImagePart, Lexical, Manifest, Modality, QueryOp, Regex,
    StructuredPart, TextPart, Vector, text_of,
)
from agentic_search.embedders.base import Embedder, cosine_scores, normalize_rows, supports

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".json", ".jsonl", ".csv", ".tsv",
                   ".html", ".xml", ".yaml", ".yml"}
IMAGE_EXTENSIONS = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".gif": "image/gif", ".webp": "image/webp"}
EMBED_BATCH = 64
SAMPLE_DISTINCT_MAX = 20
_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _is_iso_date(s: str) -> bool:
    if len(s) < 8:
        return False
    try:
        datetime.fromisoformat(s)
        return True
    except ValueError:
        return False


def infer_field_type(values: list[Any]) -> FieldType:
    vals = [v for v in values if v is not None]
    if not vals:
        return FieldType.KEYWORD
    if all(isinstance(v, bool) for v in vals):
        return FieldType.BOOL
    if all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
        return FieldType.INT
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
        return FieldType.FLOAT
    if all(isinstance(v, str) for v in vals):
        return FieldType.DATE if all(_is_iso_date(v) for v in vals[:50]) else FieldType.KEYWORD
    return FieldType.JSON


class FilesBackend:
    backend_type = "files"

    def __init__(self, name: str, root: str | Path | None = None, *,
                 documents: Iterable[Document] | None = None, embedder: Embedder | None = None,
                 glob: str = "**/*", collection: str = "files", description: str | None = None,
                 max_file_bytes: int = 2_000_000):
        if (root is None) == (documents is None):
            raise ValueError("pass exactly one of root or documents")
        self.name = name
        self.root = Path(root) if root is not None else None
        self._given = list(documents) if documents is not None else None
        self.embedder = embedder
        self.glob = glob
        self.collection = collection
        self.description = description
        self.max_file_bytes = max_file_bytes
        self._docs: dict[str, Document] = {}
        self._ids: list[str] = []
        self._token_sets: list[set[str]] = []
        self._bm25: BM25Okapi | None = None
        self._vec_ids: list[str] = []
        self._matrix: np.ndarray | None = None
        self._lock = asyncio.Lock()
        self._loaded = False

    @classmethod
    def from_documents(cls, name: str, documents: Iterable[Document], **kwargs: Any) -> FilesBackend:
        return cls(name, documents=documents, **kwargs)

    # ---- loading ------------------------------------------------------------

    async def _ensure_loaded(self) -> None:
        async with self._lock:
            if self._loaded:
                return
            docs = self._given if self._given is not None else await asyncio.to_thread(self._scan)
            self._docs = {d.doc_id: d for d in docs}
            self._ids = list(self._docs)
            corpus = [tokenize(text_of(self._docs[i].content)) for i in self._ids]
            self._token_sets = [set(toks) for toks in corpus]
            if any(corpus):
                self._bm25 = BM25Okapi(corpus)
            if self.embedder is not None:
                await self._build_vectors()
            self._loaded = True

    def _scan(self) -> list[Document]:
        assert self.root is not None
        docs: list[Document] = []
        for path in sorted(self.root.glob(self.glob)):
            if not path.is_file():
                continue
            ext = path.suffix.lower()
            stat = path.stat()
            rel = path.relative_to(self.root).as_posix()
            meta = {"path": rel, "ext": ext, "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()}
            content: list[Content]
            if ext in TEXT_EXTENSIONS:
                if stat.st_size > self.max_file_bytes:
                    continue
                content = [TextPart(text=path.read_text(encoding="utf-8", errors="replace"))]
            elif ext in IMAGE_EXTENSIONS:
                content = [ImagePart(uri=path.resolve().as_uri(), mime=IMAGE_EXTENSIONS[ext])]
            else:
                continue
            docs.append(Document(doc_id=rel, content=content, metadata=meta))
        return docs

    def _embeddable_part(self, doc: Document) -> Content | None:
        assert self.embedder is not None
        text = text_of(doc.content)
        if text and supports(self.embedder, TextPart(text=text)):
            return TextPart(text=text)
        for part in doc.content:
            if isinstance(part, ImagePart) and supports(self.embedder, part):
                return part
        return None

    async def _build_vectors(self) -> None:
        assert self.embedder is not None
        items = [(i, p) for i in self._ids if (p := self._embeddable_part(self._docs[i])) is not None]
        vectors: list[list[float]] = []
        for start in range(0, len(items), EMBED_BATCH):
            batch = items[start:start + EMBED_BATCH]
            vectors.extend(await self.embedder.embed([p for _, p in batch], "document"))
        self._vec_ids = [i for i, _ in items]
        self._matrix = normalize_rows(np.asarray(vectors, dtype=np.float32)) if vectors else None

    # ---- protocol -----------------------------------------------------------

    def capabilities(self) -> set[Capability]:
        caps = {Capability.LEXICAL, Capability.FILTER, Capability.REGEX, Capability.FETCH,
                Capability.AGGREGATE}
        if self.embedder is not None:
            caps |= {Capability.VECTOR, Capability.HYBRID}
            if Modality.IMAGE in self.embedder.modalities:
                caps.add(Capability.IMAGE_QUERY)
        return caps

    async def discover(self, detail: DiscoverDetail = "full",
                       collection: str | None = None) -> Manifest:
        await self._ensure_loaded()
        fields = [FieldSpec(name="text", type=FieldType.TEXT, searchable=True)]
        fields.extend(self._metadata_fields())
        if self.embedder is not None:
            fields.append(FieldSpec(name="embedding", type=FieldType.VECTOR,
                                    vector_dim=self.embedder.dim, vector_metric="cosine",
                                    embedder_id=self.embedder.id))
        return Manifest(
            source=self.name, backend_type=self.backend_type, capabilities=self.capabilities(),
            description=self.description,
            collections=[CollectionInfo(name=self.collection, fields=fields, count=len(self._docs))],
        )

    def _metadata_fields(self) -> list[FieldSpec]:
        values: dict[str, list[Any]] = {}
        for doc in self._docs.values():
            for k, v in doc.metadata.items():
                values.setdefault(k, []).append(v)
        out = []
        for key in sorted(values):
            ftype = infer_field_type(values[key])
            distinct = list(dict.fromkeys(
                v for v in values[key] if isinstance(v, (str, int, float, bool))))
            samples = (distinct if ftype in (FieldType.KEYWORD, FieldType.BOOL)
                       and len(distinct) <= SAMPLE_DISTINCT_MAX else None)
            out.append(FieldSpec(
                name=key, type=ftype, filterable=True,
                sortable=ftype in (FieldType.INT, FieldType.FLOAT, FieldType.DATE),
                sample_values=samples))
        return out

    async def execute(self, op: QueryOp) -> list[Hit]:
        await self._ensure_loaded()
        if op.collection not in (None, self.collection):
            raise BackendError(f"unknown collection {op.collection!r}; this source has {self.collection!r}")
        if isinstance(op, Lexical):
            ranked = self._lexical_ranking(op.text, set(self._allowed(op.filter)))
            return [self._hit(i, s) for i, s in ranked[: op.limit]]
        if isinstance(op, Vector):
            self._check_vector_field(op.field)
            ranked = self._vector_ranking(op.vector, set(self._allowed(op.filter)))
            return [self._hit(i, s) for i, s in ranked[: op.limit]]
        if isinstance(op, Hybrid):
            return self._hybrid(op)
        if isinstance(op, FilterOnly):
            return [self._hit(i) for i in self._allowed(op.filter)[: op.limit]]
        if isinstance(op, Regex):
            return self._regex(op)
        if isinstance(op, Fetch):
            return [self._hit(i) for i in op.doc_ids if i in self._docs]
        if isinstance(op, Aggregate):
            return self._aggregate(op)
        raise UnsupportedOperation(f"files backend does not support {op.type}")

    async def close(self) -> None:
        return None

    # ---- op implementations -------------------------------------------------

    def _allowed(self, f: Filter | None) -> list[str]:
        if f is None:
            return list(self._ids)
        return [i for i in self._ids if matches(f, {**self._docs[i].metadata, "doc_id": i})]

    def _hit(self, doc_id: str, score: float | None = None) -> Hit:
        d = self._docs[doc_id]
        return Hit(doc_id=doc_id, source=self.name, content=d.content, metadata=d.metadata,
                   raw_score=score)

    def _lexical_ranking(self, text: str, allowed: set[str]) -> list[tuple[str, float]]:
        query = tokenize(text)
        if self._bm25 is None or not query:
            return []
        qset = set(query)
        scores = self._bm25.get_scores(query)
        ranked = [(i, float(s)) for i, s, toks in zip(self._ids, scores, self._token_sets)
                  if i in allowed and qset & toks]
        ranked.sort(key=lambda kv: (-kv[1], kv[0]))
        return ranked

    def _check_vector_field(self, field: str) -> None:
        if self.embedder is None:
            raise BackendError("this source has no vector index")
        if field != "embedding":
            raise BackendError(f"unknown vector field {field!r}; this source has 'embedding'")

    def _vector_ranking(self, vector: list[float] | None, allowed: set[str]) -> list[tuple[str, float]]:
        if vector is None:
            raise BackendError("vector op reached the backend without an embedded query vector")
        if self._matrix is None:
            return []
        scores = cosine_scores(vector, self._matrix)
        ranked = [(i, float(s)) for i, s in zip(self._vec_ids, scores) if i in allowed]
        ranked.sort(key=lambda kv: (-kv[1], kv[0]))
        return ranked

    def _hybrid(self, op: Hybrid) -> list[Hit]:
        self._check_vector_field(op.field)
        allowed = set(self._allowed(op.filter))
        depth = max(op.limit * 5, 50)
        lexical = [i for i, _ in self._lexical_ranking(op.text, allowed)][:depth]
        semantic = [i for i, _ in self._vector_ranking(op.vector, allowed)][:depth]
        fused = rrf_merge([(lexical, op.lexical_weight), (semantic, 1.0 - op.lexical_weight)])
        return [self._hit(i, s) for i, s in fused[: op.limit]]

    def _regex(self, op: Regex) -> list[Hit]:
        try:
            pattern = re.compile(op.pattern)
        except re.error as exc:
            raise BackendError(f"invalid regex: {exc}") from exc
        fields = op.fields or ["text", "path"]
        out: list[Hit] = []
        for i in self._allowed(op.filter):
            doc = self._docs[i]
            for f in fields:
                value = text_of(doc.content) if f == "text" else doc.metadata.get(f)
                if value is not None and pattern.search(str(value)):
                    out.append(self._hit(i))
                    break
            if len(out) >= op.limit:
                break
        return out

    def _aggregate(self, op: Aggregate) -> list[Hit]:
        unsupported = [m for m in op.metrics if m != "count"]
        if unsupported:
            raise UnsupportedOperation(f"files backend only supports the 'count' metric, not {unsupported}")
        groups: dict[str, tuple[dict[str, Any], int]] = {}
        for i in self._allowed(op.filter):
            row = {g: self._docs[i].metadata.get(g) for g in op.group_by}
            key = json.dumps(row, sort_keys=True, default=str)
            groups[key] = (row, groups.get(key, (row, 0))[1] + 1)
        ranked = sorted(groups.items(), key=lambda kv: (-kv[1][1], kv[0]))[: op.limit]
        return [Hit(doc_id=f"agg:{key}", source=self.name,
                    content=[StructuredPart(data={**row, "count": n})], raw_score=float(n))
                for key, (row, n) in ranked]
```

- [ ] **Step 6: Run to verify pass**

Run: `uv run pytest tests/backends -q`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(backends): Backend protocol and FilesBackend (BM25, vector, hybrid, regex, filter, aggregate)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---
### Task 6: Model role protocols + test fakes

**Files:**
- Create: `src/agentic_search/models/base.py`, `src/agentic_search/testing.py`
- Test: `tests/models/test_base.py` (plus an empty `tests/models/__init__.py`)

**Interfaces:**
- Consumes: `Budget`, `Hit`, `ModelUsage`, `Query`, `ToolError` (Task 2).
- Produces:
  - Data types:
    - `ToolSpec(name, description, parameters: dict)` and `ToolCall(id, name, arguments: dict)`
    - `Judgment(key, p_relevant ∈ [0,1], rationale)`
    - `Action` enum (CONTINUE, REFINE, BROADEN, SWITCH_SOURCE, STOP) and `Decision(action, confidence, note, usage)`
    - `TurnSummary(turn, n_calls, n_errors, n_new, n_new_relevant: int|None, top_keys)`
    - `PlannerView(question, turn, manifest_summary, digest, directive, errors, max_calls)`
    - `ControllerView(question, turn, history, digest, total_relevant, budget_remaining)`
    - `PlanResult(calls, usage, note)`, `JudgeResult(judgments, usage)`, `DelegateResult(ranked_keys, usage, note)`
  - Protocols:
    - `ToolRuntime`: `async call(calls) -> list[str]`.
    - `Driver`: `id`, `supports_images`, `async plan(view, tools) -> PlanResult`, and `async run_delegate(question, tools, runtime, budget, context="") -> DelegateResult`.
    - `Decider`: `id`, `async judge(question, hits) -> JudgeResult`, `async decide(view) -> Decision`. Either method may raise `NotImplementedError`.
  - `agentic_search.testing`: `call(name, id=None, **args)`, `ScriptedDriver`, `EchoDriver`, `KeywordJudge`, `ScriptedController`, `FailingDecider`.

- [ ] **Step 1: Write the failing tests**

`tests/models/test_base.py`:
```python
import pytest
from pydantic import ValidationError

from agentic_search.core.types import Budget, Hit, Query, TextPart
from agentic_search.models.base import (
    Action, Decider, Decision, Driver, Judgment, PlannerView,
)
from agentic_search.testing import (
    EchoDriver, FailingDecider, KeywordJudge, ScriptedController, ScriptedDriver, call,
)


def view(turn=0):
    return PlannerView(question=Query.of("q"), turn=turn, manifest_summary="", digest="")


def test_value_bounds():
    with pytest.raises(ValidationError):
        Judgment(key="k", p_relevant=1.5)
    with pytest.raises(ValidationError):
        Decision(action=Action.STOP, confidence=-0.1)


def test_call_helper_assigns_ids():
    a, b = call("lexical_search", text="x"), call("lexical_search", text="y")
    assert a.id != b.id and a.arguments == {"text": "x"}


async def test_scripted_driver():
    d = ScriptedDriver([[call("x")]])
    assert isinstance(d, Driver)
    assert len((await d.plan(view(), [])).calls) == 1
    assert (await d.plan(view(1), [])).calls == []
    assert len(d.views) == 2


async def test_scripted_driver_delegate():
    class Runtime:
        async def call(self, calls):
            return [f"out:{c.name}" for c in calls]

    d = ScriptedDriver(delegate_calls=[[call("a")]], delegate_keys=["s:1"])
    res = await d.run_delegate(Query.of("q"), [], Runtime(), Budget(), context="ctx")
    assert res.ranked_keys == ["s:1"] and d.delegate_outputs == [["out:a"]]
    assert d.delegate_context == "ctx"


async def test_echo_driver():
    d = EchoDriver("docs", limit=5)
    [c] = (await d.plan(view(), [])).calls
    assert c.name == "lexical_search" and c.arguments == {"source": "docs", "text": "q", "limit": 5}
    assert (await d.plan(view(1), [])).calls == []


async def test_keyword_judge_and_controllers():
    j = KeywordJudge(["fever"])
    assert isinstance(j, Decider)
    hits = [Hit(doc_id="1", source="s", content=[TextPart(text="Fever high")]),
            Hit(doc_id="2", source="s", content=[TextPart(text="nothing")])]
    res = await j.judge(Query.of("q"), hits)
    assert [x.p_relevant for x in res.judgments] == [1.0, 0.0]
    with pytest.raises(NotImplementedError):
        await j.decide(None)
    c = ScriptedController([Action.CONTINUE])
    assert (await c.decide(None)).action is Action.CONTINUE
    assert (await c.decide(None)).action is Action.STOP
    with pytest.raises(RuntimeError):
        await FailingDecider().judge(Query.of("q"), hits)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/models/test_base.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement protocols**

`src/agentic_search/models/base.py`:
```python
"""Model role protocols. Drivers plan and call tools; deciders make typed judgments."""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agentic_search.core.types import Budget, Hit, ModelUsage, Query, ToolError


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Judgment(BaseModel):
    key: str
    p_relevant: float = Field(ge=0.0, le=1.0)
    rationale: str | None = None


class Action(str, Enum):
    CONTINUE = "continue"
    REFINE = "refine"
    BROADEN = "broaden"
    SWITCH_SOURCE = "switch_source"
    STOP = "stop"


class Decision(BaseModel):
    action: Action
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    note: str | None = None
    usage: ModelUsage = Field(default_factory=ModelUsage)


class TurnSummary(BaseModel):
    turn: int
    n_calls: int
    n_errors: int
    n_new: int
    n_new_relevant: int | None
    top_keys: list[str] = Field(default_factory=list)


class PlannerView(BaseModel):
    question: Query
    turn: int
    manifest_summary: str
    digest: str
    directive: Decision | None = None
    errors: list[ToolError] = Field(default_factory=list)
    max_calls: int = 8


class ControllerView(BaseModel):
    question: Query
    turn: int
    history: list[TurnSummary]
    digest: str
    total_relevant: int | None
    budget_remaining: dict[str, float | int | None]


class PlanResult(BaseModel):
    calls: list[ToolCall]
    usage: ModelUsage = Field(default_factory=ModelUsage)
    note: str | None = None


class JudgeResult(BaseModel):
    judgments: list[Judgment]
    usage: ModelUsage = Field(default_factory=ModelUsage)


class DelegateResult(BaseModel):
    ranked_keys: list[str]
    usage: ModelUsage = Field(default_factory=ModelUsage)
    note: str | None = None


class ToolRuntime(Protocol):
    """Executes tool calls for a delegate-mode driver; returns one text result per call."""

    async def call(self, calls: list[ToolCall]) -> list[str]: ...


@runtime_checkable
class Driver(Protocol):
    id: str
    supports_images: bool

    async def plan(self, view: PlannerView, tools: list[ToolSpec]) -> PlanResult: ...

    async def run_delegate(self, question: Query, tools: list[ToolSpec], runtime: ToolRuntime,
                           budget: Budget, context: str = "") -> DelegateResult: ...


@runtime_checkable
class Decider(Protocol):
    """Implement judge, decide, or both; raise NotImplementedError for the one you don't support."""

    id: str

    async def judge(self, question: Query, hits: list[Hit]) -> JudgeResult: ...

    async def decide(self, view: ControllerView) -> Decision: ...
```

- [ ] **Step 4: Implement fakes**

`src/agentic_search/testing.py`:
```python
"""Deterministic fakes for tests and offline demos. No network, no models."""

from __future__ import annotations

import itertools
from typing import Any

from agentic_search.core.types import Budget, Hit, ModelUsage, Query
from agentic_search.models.base import (
    Action, ControllerView, Decision, DelegateResult, JudgeResult, Judgment, PlannerView,
    PlanResult, ToolCall, ToolRuntime, ToolSpec,
)

_ids = itertools.count(1)


def call(name: str, id: str | None = None, **arguments: Any) -> ToolCall:
    return ToolCall(id=id or f"call{next(_ids)}", name=name, arguments=arguments)


class ScriptedDriver:
    """plan() returns turns[i] on the i-th call, then nothing. run_delegate() replays delegate_calls."""

    def __init__(self, turns: list[list[ToolCall]] | None = None, *,
                 delegate_calls: list[list[ToolCall]] | None = None,
                 delegate_keys: list[str] | None = None, id: str = "scripted-driver",
                 supports_images: bool = False, usage_per_plan: ModelUsage | None = None):
        self.turns = turns or []
        self.delegate_calls = delegate_calls or []
        self.delegate_keys = delegate_keys or []
        self.id = id
        self.supports_images = supports_images
        self.usage_per_plan = usage_per_plan or ModelUsage()
        self.views: list[PlannerView] = []
        self.tools_seen: list[list[ToolSpec]] = []
        self.delegate_outputs: list[list[str]] = []
        self.delegate_context: str | None = None

    async def plan(self, view: PlannerView, tools: list[ToolSpec]) -> PlanResult:
        self.views.append(view)
        self.tools_seen.append(tools)
        i = len(self.views) - 1
        calls = list(self.turns[i]) if i < len(self.turns) else []
        return PlanResult(calls=calls, usage=self.usage_per_plan)

    async def run_delegate(self, question: Query, tools: list[ToolSpec], runtime: ToolRuntime,
                           budget: Budget, context: str = "") -> DelegateResult:
        self.tools_seen.append(tools)
        self.delegate_context = context
        for calls in self.delegate_calls:
            self.delegate_outputs.append(await runtime.call(calls))
        return DelegateResult(ranked_keys=list(self.delegate_keys))


class EchoDriver:
    """Turn 0: one lexical search of the question text. Later turns: nothing. A BM25 baseline."""

    def __init__(self, source: str, *, limit: int = 100, id: str = "echo-driver"):
        self.source = source
        self.limit = limit
        self.id = id
        self.supports_images = False

    async def plan(self, view: PlannerView, tools: list[ToolSpec]) -> PlanResult:
        if view.turn > 0:
            return PlanResult(calls=[])
        return PlanResult(calls=[call("lexical_search", source=self.source,
                                      text=view.question.as_text(), limit=self.limit)])

    async def run_delegate(self, question: Query, tools: list[ToolSpec], runtime: ToolRuntime,
                           budget: Budget, context: str = "") -> DelegateResult:
        raise NotImplementedError


class KeywordJudge:
    """p_relevant = 1.0 if any keyword appears in the hit text, else 0.0."""

    def __init__(self, keywords: list[str], *, id: str = "keyword-judge"):
        self.keywords = [k.lower() for k in keywords]
        self.id = id

    async def judge(self, question: Query, hits: list[Hit]) -> JudgeResult:
        out = []
        for h in hits:
            found = any(k in h.snippet(100_000).lower() for k in self.keywords)
            out.append(Judgment(key=h.key, p_relevant=1.0 if found else 0.0,
                                rationale="keyword match" if found else "no keyword"))
        return JudgeResult(judgments=out)

    async def decide(self, view: ControllerView) -> Decision:
        raise NotImplementedError


class ScriptedController:
    """decide() pops the next scripted action, then STOP forever."""

    def __init__(self, actions: list[Action], *, id: str = "scripted-controller"):
        self.actions = list(actions)
        self.id = id
        self.views: list[ControllerView] = []

    async def judge(self, question: Query, hits: list[Hit]) -> JudgeResult:
        raise NotImplementedError

    async def decide(self, view: ControllerView) -> Decision:
        self.views.append(view)
        action = self.actions.pop(0) if self.actions else Action.STOP
        return Decision(action=action, note="scripted")


class FailingDecider:
    id = "failing-decider"

    async def judge(self, question: Query, hits: list[Hit]) -> JudgeResult:
        raise RuntimeError("judge exploded")

    async def decide(self, view: ControllerView) -> Decision:
        raise RuntimeError("decide exploded")
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/models/test_base.py -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(models): driver/decider protocols and deterministic fakes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Search state + hooks

**Files:**
- Create: `src/agentic_search/core/state.py`, `src/agentic_search/core/hooks.py`
- Test: `tests/core/test_state.py`

**Interfaces:**
- Consumes: Tasks 2 and 6.
- Produces:
  - `TraceEvent(type, turn, at_ms, duration_ms, data)`.
  - `Trace`: `.events`, `.add(type, turn, *, duration_ms=None, **data) -> TraceEvent`, `.set_listener(fn)`, `.of_type(type) -> list[TraceEvent]`.
  - `Usage(input_tokens, output_tokens, cost_usd, tool_calls, turns)` with `.add_model(ModelUsage)` and `.total_tokens`.
  - `Candidate` dataclass: `hit`, `first_turn`, `p_relevant`, `rationale`, `judged`, `unjudged_reason`.
  - `CandidatePool(rrf_k=60)`:
    - `.add(hits, *, turn, call_id) -> list[str]` returns the keys that are new to the pool.
    - `pool[key]`, `key in pool`, `len(pool)` and `.candidates()`.
    - `.rrf(key)`, `.scores(judge_weight=0.9) -> dict[str, float]`, `.ranked(judge_weight=0.9) -> list[tuple[Candidate, float]]`.
  - `SearchState` dataclass: `question`, `manifests`, `budget`, `pool`, `trace`, `usage`, `turn`, `history`, `digest`, `last_decision`, `last_errors`, `started`, and `.elapsed()`.
  - `Hooks`: `async before_model_call(model_id, payload) -> payload` and `on_trace_event(event)`.
  - `SourcePolicy(allowed: dict[str, set[str]] | None)`: `.allows(source, model_id)` and `.redact(hits, model_id)`.

- [ ] **Step 1: Write the failing tests**

`tests/core/test_state.py`:
```python
from agentic_search.core.hooks import Hooks, SourcePolicy
from agentic_search.core.state import CandidatePool, SearchState, Trace, Usage
from agentic_search.core.types import Budget, Hit, ModelUsage, Query, TextPart


def hit(doc_id, source="s", text="t"):
    return Hit(doc_id=doc_id, source=source, content=[TextPart(text=text)], metadata={"m": 1})


def test_pool_add_dedupes_and_tracks_provenance():
    pool = CandidatePool()
    assert pool.add([hit("a"), hit("b")], turn=0, call_id="c1") == ["s:a", "s:b"]
    assert pool.add([hit("b"), hit("c")], turn=1, call_id="c2") == ["s:c"]
    assert len(pool) == 3 and "s:b" in pool
    prov = pool["s:b"].hit.provenance
    assert [(p.call_id, p.rank) for p in prov] == [("c1", 1), ("c2", 0)]
    assert pool["s:c"].first_turn == 1


def test_pool_rrf_and_blended_scores():
    pool = CandidatePool(rrf_k=0)
    pool.add([hit("a"), hit("b")], turn=0, call_id="c1")
    pool.add([hit("b")], turn=0, call_id="c2")
    assert pool.rrf("s:b") == 1 / 2 + 1 / 1
    scores = pool.scores(judge_weight=0.9)
    assert scores["s:b"] == 1.0  # unjudged: normalized rrf
    pool["s:a"].judged, pool["s:a"].p_relevant = True, 1.0
    pool["s:b"].judged, pool["s:b"].p_relevant = True, 0.0
    ranked = pool.ranked(judge_weight=0.9)
    assert [c.hit.doc_id for c, _ in ranked] == ["a", "b"]


def test_trace_listener_and_json():
    seen = []
    t = Trace()
    t.set_listener(seen.append)
    ev = t.add("plan", 0, duration_ms=1.5, n_calls=2)
    assert seen == [ev] and ev.data == {"n_calls": 2}
    assert t.of_type("plan") == [ev]
    assert '"plan"' in t.model_dump_json()


def test_usage_add_model():
    u = Usage()
    u.add_model(ModelUsage(input_tokens=3, output_tokens=4, cost_usd=0.1))
    assert u.total_tokens == 7 and u.cost_usd == 0.1


def test_search_state_defaults():
    s = SearchState(question=Query.of("q"), manifests={}, budget=Budget())
    assert s.turn == 0 and len(s.pool) == 0 and s.elapsed() >= 0
    assert s.digest.startswith("No searches")


async def test_default_hooks_pass_through():
    h = Hooks()
    assert await h.before_model_call("m", {"x": 1}) == {"x": 1}
    assert h.on_trace_event(None) is None


def test_source_policy_redacts():
    p = SourcePolicy({"phi": {"local-model"}})
    assert p.allows("public", "any") and p.allows("phi", "local-model")
    assert not p.allows("phi", "cloud-model")
    out = p.redact([hit("a", "phi"), hit("b", "public")], "cloud-model")
    assert out[0].content == [] and out[0].metadata["_redacted"] is True
    assert out[1].content != []
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/core/test_state.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement state**

`src/agentic_search/core/state.py`:
```python
"""Per-search mutable state: candidate pool, trace, usage."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, Field, PrivateAttr

from agentic_search.core.types import Budget, Hit, Manifest, ModelUsage, OpRef, Query, ToolError
from agentic_search.models.base import Decision, TurnSummary


class TraceEvent(BaseModel):
    type: str
    turn: int
    at_ms: float
    duration_ms: float | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class Trace(BaseModel):
    events: list[TraceEvent] = Field(default_factory=list)
    _t0: float = PrivateAttr(default_factory=time.monotonic)
    _listener: Callable[[TraceEvent], None] | None = PrivateAttr(default=None)

    def set_listener(self, fn: Callable[[TraceEvent], None] | None) -> None:
        self._listener = fn

    def add(self, type: str, turn: int, *, duration_ms: float | None = None,
            **data: Any) -> TraceEvent:
        event = TraceEvent(type=type, turn=turn, at_ms=(time.monotonic() - self._t0) * 1000,
                           duration_ms=duration_ms, data=data)
        self.events.append(event)
        if self._listener is not None:
            self._listener(event)
        return event

    def of_type(self, type: str) -> list[TraceEvent]:
        return [e for e in self.events if e.type == type]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    tool_calls: int = 0
    turns: int = 0

    def add_model(self, usage: ModelUsage) -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cost_usd += usage.cost_usd

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Candidate:
    hit: Hit
    first_turn: int
    p_relevant: float | None = None
    rationale: str | None = None
    judged: bool = False
    unjudged_reason: str | None = None


class CandidatePool:
    """Every hit seen during one search, de-duplicated by key, with provenance for RRF."""

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self._c: dict[str, Candidate] = {}

    def add(self, hits: list[Hit], *, turn: int, call_id: str) -> list[str]:
        new: list[str] = []
        for rank, h in enumerate(hits):
            ref = OpRef(turn=turn, call_id=call_id, rank=rank)
            existing = self._c.get(h.key)
            if existing is not None:
                existing.hit.provenance.append(ref)
                continue
            self._c[h.key] = Candidate(hit=h.model_copy(update={"provenance": [*h.provenance, ref]}),
                                       first_turn=turn)
            new.append(h.key)
        return new

    def __getitem__(self, key: str) -> Candidate:
        return self._c[key]

    def __contains__(self, key: object) -> bool:
        return key in self._c

    def __len__(self) -> int:
        return len(self._c)

    def candidates(self) -> list[Candidate]:
        return list(self._c.values())

    def rrf(self, key: str) -> float:
        return sum(1.0 / (self.rrf_k + ref.rank + 1) for ref in self._c[key].hit.provenance)

    def scores(self, judge_weight: float = 0.9) -> dict[str, float]:
        """Judged: w*p + (1-w)*rrf_norm. Unjudged: rrf_norm (fail-open keeps the fused score)."""
        rrf = {k: self.rrf(k) for k in self._c}
        top = max(rrf.values(), default=0.0) or 1.0
        out: dict[str, float] = {}
        for k, c in self._c.items():
            norm = rrf[k] / top
            if c.judged and c.p_relevant is not None:
                out[k] = judge_weight * c.p_relevant + (1.0 - judge_weight) * norm
            else:
                out[k] = norm
        return out

    def ranked(self, judge_weight: float = 0.9) -> list[tuple[Candidate, float]]:
        scores = self.scores(judge_weight)
        return sorted(((c, scores[k]) for k, c in self._c.items()),
                      key=lambda cs: (-cs[1], cs[0].hit.key))


@dataclass
class SearchState:
    question: Query
    manifests: dict[str, Manifest]
    budget: Budget
    pool: CandidatePool = field(default_factory=CandidatePool)
    trace: Trace = field(default_factory=Trace)
    usage: Usage = field(default_factory=Usage)
    turn: int = 0
    history: list[TurnSummary] = field(default_factory=list)
    digest: str = "No searches have been run yet."
    last_decision: Decision | None = None
    last_errors: list[ToolError] = field(default_factory=list)
    started: float = field(default_factory=time.monotonic)

    def elapsed(self) -> float:
        return time.monotonic() - self.started
```

- [ ] **Step 4: Implement hooks**

`src/agentic_search/core/hooks.py`:
```python
"""Extension points for redaction, audit, and per-source model access policy."""

from __future__ import annotations

from typing import Any

from agentic_search.core.state import TraceEvent
from agentic_search.core.types import Hit


class Hooks:
    """Subclass to redact, audit, or block. Every payload bound for an external model or
    embedder passes through before_model_call; the returned value is what gets sent."""

    async def before_model_call(self, model_id: str, payload: Any) -> Any:
        return payload

    def on_trace_event(self, event: TraceEvent) -> None:
        return None


class SourcePolicy:
    """source -> model ids allowed to see that source's raw content. Unlisted sources are open."""

    def __init__(self, allowed: dict[str, set[str]] | None = None):
        self._allowed = {k: set(v) for k, v in (allowed or {}).items()}

    def allows(self, source: str, model_id: str) -> bool:
        allowed = self._allowed.get(source)
        return allowed is None or model_id in allowed

    def redact(self, hits: list[Hit], model_id: str) -> list[Hit]:
        return [h if self.allows(h.source, model_id)
                else h.model_copy(update={"content": [], "metadata": {**h.metadata, "_redacted": True}})
                for h in hits]
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/core -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(core): candidate pool, trace, usage, search state, hooks and source policy

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Tool specs and call parsing

**Files:**
- Create: `src/agentic_search/roles/tools.py`
- Test: `tests/roles/test_tools.py` (plus an empty `tests/roles/__init__.py`)

**Interfaces:**
- Consumes: `QUERY_OP_ADAPTER`, `Capability`, `ImagePart`, `Manifest`, `Query` (Task 2), and `ToolCall`, `ToolSpec` (Task 6).
- Produces:
  - `DISCOVER_TOOL = "discover"` and `TOOL_OPS: dict[str, tuple[str, Capability]]`.
  - `build_tool_specs(manifests) -> list[ToolSpec]`. It always includes `discover`. Every other tool is included only if some source has its capability, and its `source` enum lists only those sources.
  - `DiscoverRequest(source, collection=None)`.
  - `parse_call(call, question) -> QueryOp | DiscoverRequest`. It raises `ValueError` with a message fit to show the model.

- [ ] **Step 1: Write the failing tests**

`tests/roles/test_tools.py`:
```python
import pytest

from agentic_search.core.types import (
    Capability, ImagePart, Lexical, Manifest, Query, TextPart, Vector,
)
from agentic_search.roles.tools import DiscoverRequest, build_tool_specs, parse_call
from agentic_search.testing import call


def manifest(name, caps):
    return Manifest(source=name, backend_type="x", capabilities=set(caps))


def test_specs_follow_capabilities():
    specs = {s.name: s for s in build_tool_specs({
        "a": manifest("a", [Capability.LEXICAL, Capability.FILTER]),
        "b": manifest("b", [Capability.VECTOR]),
    })}
    assert set(specs) == {"discover", "lexical_search", "filter_search", "vector_search"}
    assert specs["lexical_search"].parameters["properties"]["source"]["enum"] == ["a"]
    assert specs["vector_search"].parameters["properties"]["source"]["enum"] == ["b"]
    assert specs["discover"].parameters["properties"]["source"]["enum"] == ["a", "b"]
    assert specs["lexical_search"].parameters["required"] == ["source", "text"]


def test_parse_lexical_and_discover():
    q = Query.of("q")
    op = parse_call(call("lexical_search", source="a", text="x", type="vector"), q)
    assert isinstance(op, Lexical) and op.text == "x"
    d = parse_call(call("discover", source="a"), q)
    assert d == DiscoverRequest(source="a")


def test_parse_vector_strips_model_supplied_vector():
    op = parse_call(call("vector_search", source="a", field="e", hyde_text="h", vector=[1.0]),
                    Query.of("q"))
    assert isinstance(op, Vector) and op.vector is None


def test_parse_vector_uses_question_image():
    img = ImagePart(data=b"png")
    q = Query(content=[TextPart(text="find similar"), img])
    op = parse_call(call("vector_search", source="a", field="e", use_question_image=True), q)
    assert op.content == img and op.hyde_text is None
    with pytest.raises(ValueError, match="no image"):
        parse_call(call("vector_search", source="a", field="e", use_question_image=True),
                   Query.of("q"))


def test_parse_errors_are_value_errors():
    q = Query.of("q")
    with pytest.raises(ValueError, match="unknown tool"):
        parse_call(call("drop_table", source="a"), q)
    with pytest.raises(ValueError):
        parse_call(call("lexical_search", source="a"), q)  # missing text
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_call(call("lexical_search", _invalid_json="{oops"), q)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/roles/test_tools.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`src/agentic_search/roles/tools.py`:
```python
"""Tool specs offered to drivers, and parsing tool calls back into query ops."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agentic_search.core.types import QUERY_OP_ADAPTER, Capability, Manifest, Query, QueryOp
from agentic_search.models.base import ToolCall, ToolSpec

DISCOVER_TOOL = "discover"

FILTER_DOC = (
    'Filter JSON. Leaves: {"op":"eq","field":F,"value":V} | {"op":"in","field":F,"values":[...]} | '
    '{"op":"range","field":F,"gte"|"gt"|"lte"|"lt":V} | {"op":"exists","field":F} | '
    '{"op":"contains","field":F,"value":S}. Combine with {"op":"and"|"or","clauses":[...]} or '
    '{"op":"not","clause":{...}}. Use only filterable fields from the source description.'
)

TOOL_OPS: dict[str, tuple[str, Capability]] = {
    "lexical_search": ("lexical", Capability.LEXICAL),
    "vector_search": ("vector", Capability.VECTOR),
    "hybrid_search": ("hybrid", Capability.HYBRID),
    "filter_search": ("filter", Capability.FILTER),
    "regex_search": ("regex", Capability.REGEX),
    "traverse": ("traverse", Capability.TRAVERSE),
    "aggregate": ("aggregate", Capability.AGGREGATE),
    "fetch": ("fetch", Capability.FETCH),
    "native_query": ("native", Capability.NATIVE),
}

_DESCRIPTIONS = {
    "lexical_search": "Keyword/BM25 full-text search. Best for exact terms, names, codes, rare words.",
    "vector_search": ("Semantic nearest-neighbor search on a vector field. Pass hyde_text: a short "
                      "hypothetical passage written the way a relevant document would read. Set "
                      "use_question_image=true to search with the image attached to the question."),
    "hybrid_search": "Keyword + semantic search on a vector field, fused by rank.",
    "filter_search": "Return items matching a structured filter (no ranking).",
    "regex_search": "Regular-expression match over text or named fields.",
    "traverse": "Graph traversal from nodes matching `start` along relationship types.",
    "aggregate": "Group matching items by fields and compute metrics (e.g. count) to learn the data.",
    "fetch": "Fetch full items by id.",
    "native_query": ("Read-only query in the source's native language (SQL, Cypher, DSL). Use only "
                     "when the other tools cannot express the query."),
}

_STR = {"type": "string"}
_STR_LIST = {"type": "array", "items": {"type": "string"}}

_EXTRA: dict[str, tuple[dict[str, Any], list[str]]] = {
    "lexical_search": ({"text": _STR, "fields": _STR_LIST}, ["text"]),
    "vector_search": ({"field": {"type": "string", "description": "A vector field name."},
                       "hyde_text": _STR, "use_question_image": {"type": "boolean"}}, ["field"]),
    "hybrid_search": ({"text": _STR, "field": _STR,
                       "lexical_weight": {"type": "number", "minimum": 0, "maximum": 1}},
                      ["text", "field"]),
    "filter_search": ({}, ["filter"]),
    "regex_search": ({"pattern": _STR, "fields": _STR_LIST}, ["pattern"]),
    "traverse": ({"start": {"type": "object", "description": "Selects start nodes. " + FILTER_DOC},
                  "rel_types": _STR_LIST,
                  "direction": {"type": "string", "enum": ["out", "in", "both"]},
                  "depth": {"type": "integer", "minimum": 1, "maximum": 5},
                  "target_label": _STR}, ["start"]),
    "aggregate": ({"group_by": _STR_LIST,
                   "metrics": {**_STR_LIST, "description": "e.g. [\"count\"]"}}, ["group_by"]),
    "fetch": ({"doc_ids": _STR_LIST}, ["doc_ids"]),
    "native_query": ({"dialect": {"type": "string", "description": "sql | cypher | opensearch_dsl"},
                      "query": _STR}, ["dialect", "query"]),
}


def _base_props(sources: list[str]) -> dict[str, Any]:
    return {
        "source": {"type": "string", "enum": sources},
        "collection": {"type": "string", "description": (
            "Collection/table/index/label from the source description. Optional when the source "
            "has exactly one collection.")},
        "filter": {"type": "object", "description": FILTER_DOC},
        "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
    }


def build_tool_specs(manifests: dict[str, Manifest]) -> list[ToolSpec]:
    specs = [ToolSpec(
        name=DISCOVER_TOOL,
        description=("Show the detailed schema of a source, or of one collection: fields, types, "
                     "sample values, vector fields and their embedders."),
        parameters={"type": "object",
                    "properties": {"source": {"type": "string", "enum": sorted(manifests)},
                                   "collection": _STR},
                    "required": ["source"]},
    )]
    for tool, (_, cap) in TOOL_OPS.items():
        sources = sorted(s for s, m in manifests.items() if cap in m.capabilities)
        if not sources:
            continue
        extra, required = _EXTRA[tool]
        specs.append(ToolSpec(name=tool, description=_DESCRIPTIONS[tool], parameters={
            "type": "object",
            "properties": {**_base_props(sources), **extra},
            "required": ["source", *required],
        }))
    return specs


class DiscoverRequest(BaseModel):
    source: str
    collection: str | None = None


def parse_call(call: ToolCall, question: Query) -> QueryOp | DiscoverRequest:
    """Turn a model's tool call into a validated op. Raises ValueError on bad calls."""
    args = dict(call.arguments)
    if "_invalid_json" in args:
        raise ValueError("tool arguments were not valid JSON")
    if call.name == DISCOVER_TOOL:
        return DiscoverRequest.model_validate(args)
    if call.name not in TOOL_OPS:
        raise ValueError(f"unknown tool {call.name!r}")
    op_type = TOOL_OPS[call.name][0]
    args.pop("vector", None)  # only the executor sets query vectors
    args.pop("type", None)
    if op_type == "vector" and args.pop("use_question_image", False):
        images = question.images()
        if not images:
            raise ValueError("use_question_image was set but the question has no image")
        args["content"] = images[0]
        args.pop("hyde_text", None)
    return QUERY_OP_ADAPTER.validate_python({**args, "type": op_type})
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/roles/test_tools.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(roles): capability-driven tool specs and call parsing

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---
### Task 9: Executor

**Files:**
- Create: `src/agentic_search/roles/executor.py`
- Test: `tests/roles/test_executor.py`

**Interfaces:**
- Consumes: `Backend` (Task 5), `filter_fields` (Task 3), `EmbedderRegistry` (Task 4), `Hooks`/`SourcePolicy`/`CandidatePool`/`Trace` (Task 7), and `parse_call`/`DiscoverRequest` (Task 8).
- Produces:
  - `ExecResult` dataclass: `new_keys: list[str]`, `errors: list[ToolError]`, `outputs: dict[call_id, str]`, `hits_per_call: dict[call_id, list[key]]`.
  - `Executor(backends: dict[str, Backend], manifests: dict[str, Manifest], embedders: EmbedderRegistry, *, hooks=None, policy=None, call_timeout=30.0, per_source_concurrency=4, max_limit=100, output_hits=10, snippet_chars=240)`.
  - `.validate(op) -> tuple[kind, message] | None`.
  - `async .run(calls, *, question, turn, pool, trace, model_id=None) -> ExecResult`. It never raises for bad calls.
  - Trace events: `tool_call`, `tool_result` and `tool_error`.

- [ ] **Step 1: Write the failing tests**

`tests/roles/test_executor.py`:
```python
import asyncio

import pytest

from agentic_search.core.hooks import Hooks, SourcePolicy
from agentic_search.core.state import CandidatePool, Trace
from agentic_search.core.types import Capability, CollectionInfo, Manifest, Query
from agentic_search.embedders.base import EmbedderRegistry
from agentic_search.embedders.local import HashEmbedder
from agentic_search.roles.executor import Executor
from agentic_search.testing import call

Q = Query.of("headache")


class StubBackend:
    backend_type = "stub"

    def __init__(self, name, behavior):
        self.name, self.behavior = name, behavior

    def capabilities(self):
        return {Capability.LEXICAL}

    async def discover(self, detail="full", collection=None):
        return Manifest(source=self.name, backend_type="stub", capabilities=self.capabilities(),
                        collections=[CollectionInfo(name="c")])

    async def execute(self, op):
        if self.behavior == "raise":
            raise RuntimeError("db down")
        await asyncio.sleep(1)
        return []

    async def close(self):
        pass


class RecordingHooks(Hooks):
    def __init__(self):
        self.model_ids = []

    async def before_model_call(self, model_id, payload):
        self.model_ids.append(model_id)
        return payload


@pytest.fixture
async def ex(docs_backend):
    m = await docs_backend.discover()
    return Executor({"docs": docs_backend}, {"docs": m}, EmbedderRegistry([HashEmbedder()]),
                    hooks=RecordingHooks())


async def run(executor, *calls, model_id=None):
    pool, trace = CandidatePool(), Trace()
    res = await executor.run(list(calls), question=Q, turn=0, pool=pool, trace=trace,
                             model_id=model_id)
    return res, pool, trace


async def test_lexical_pools_hits(ex):
    c = call("lexical_search", source="docs", text="headache")
    res, pool, trace = await run(ex, c)
    assert set(res.new_keys) == {"docs:d1", "docs:d4"} and len(pool) == 2
    assert res.hits_per_call[c.id][0] in {"docs:d1", "docs:d4"}
    assert res.outputs[c.id].startswith("2 hits (2 new)")
    assert [e.type for e in trace.events] == ["tool_call", "tool_result"]


async def test_second_call_only_reports_new(ex):
    a = call("lexical_search", source="docs", text="headache")
    b = call("lexical_search", source="docs", text="fever pain")
    res, _, _ = await run(ex, a, b)
    assert sorted(res.new_keys) == ["docs:d1", "docs:d2", "docs:d4"]


async def test_validation_errors_do_not_raise(ex):
    bad_source = call("lexical_search", source="nope", text="x")
    bad_field = call("filter_search", source="docs", filter={"op": "eq", "field": "colour", "value": 1})
    bad_args = call("lexical_search", source="docs")
    res, pool, trace = await run(ex, bad_source, bad_field, bad_args)
    assert len(pool) == 0 and len(res.errors) == 3
    assert "unknown source" in res.errors[0].message
    assert "colour" in res.errors[1].message and "known fields" in res.errors[1].message
    assert res.errors[2].kind == "validation"
    assert res.outputs[bad_source.id].startswith("ERROR [validation]")
    assert len(trace.of_type("tool_error")) == 3


async def test_vector_embeds_via_bound_embedder_and_hooks(ex):
    c = call("vector_search", source="docs", field="embedding", hyde_text="headache and fever")
    res, _, _ = await run(ex, c)
    assert res.hits_per_call[c.id][0] in {"docs:d1", "docs:d4"}
    assert ex.hooks.model_ids == ["hash"]


async def test_vector_refuses_unregistered_embedder(docs_backend):
    m = await docs_backend.discover()
    ex = Executor({"docs": docs_backend}, {"docs": m}, EmbedderRegistry())
    res, _, _ = await run(ex, call("vector_search", source="docs", field="embedding", hyde_text="x"))
    assert res.errors[0].kind == "embedder" and "refusing" in res.errors[0].message


async def test_limit_is_clamped(docs_backend):
    m = await docs_backend.discover()
    ex = Executor({"docs": docs_backend}, {"docs": m}, EmbedderRegistry(), max_limit=1)
    c = call("lexical_search", source="docs", text="headache", limit=50)
    res, _, _ = await run(ex, c)
    assert len(res.hits_per_call[c.id]) == 1


async def test_backend_error_and_timeout():
    boom, slow = StubBackend("boom", "raise"), StubBackend("slow", "sleep")
    manifests = {b.name: await b.discover() for b in (boom, slow)}
    ex = Executor({"boom": boom, "slow": slow}, manifests, EmbedderRegistry(), call_timeout=0.05)
    res, _, _ = await run(ex, call("lexical_search", source="boom", text="x"),
                          call("lexical_search", source="slow", text="x"))
    assert [e.kind for e in res.errors] == ["backend", "timeout"]
    assert "db down" in res.errors[0].message


async def test_discover_renders_manifest(ex):
    c1 = call("discover", source="docs")
    c2 = call("discover", source="docs", collection="files")
    res, _, _ = await run(ex, c1, c2)
    assert "`docs`" in res.outputs[c1.id]
    assert '"embedder_id": "hash"' in res.outputs[c2.id]


async def test_policy_withholds_content_in_outputs(docs_backend):
    m = await docs_backend.discover()
    ex = Executor({"docs": docs_backend}, {"docs": m}, EmbedderRegistry(),
                  policy=SourcePolicy({"docs": {"local"}}))
    c = call("lexical_search", source="docs", text="headache")
    res, _, _ = await run(ex, c, model_id="cloud")
    assert "withheld" in res.outputs[c.id] and "Aspirin" not in res.outputs[c.id]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/roles/test_executor.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`src/agentic_search/roles/executor.py`:
```python
"""Validates tool calls against manifests, embeds queries, runs ops in parallel, pools hits."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from agentic_search.backends.base import Backend
from agentic_search.backends.filters import filter_fields
from agentic_search.core.hooks import Hooks, SourcePolicy
from agentic_search.core.state import CandidatePool, Trace
from agentic_search.core.types import (
    Aggregate, FieldType, Hit, Hybrid, Lexical, Manifest, Query, QueryOp, Regex, TextPart,
    ToolError, Vector, modality_of, required_capabilities,
)
from agentic_search.embedders.base import EmbedderRegistry
from agentic_search.models.base import ToolCall
from agentic_search.roles.tools import DiscoverRequest, parse_call

_COLLECTION_SCOPED = {"lexical", "vector", "hybrid", "filter", "regex", "aggregate"}
MAX_DISCOVER_CHARS = 8000


@dataclass
class ExecResult:
    new_keys: list[str] = field(default_factory=list)
    errors: list[ToolError] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    hits_per_call: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class _Outcome:
    hits: list[Hit] = field(default_factory=list)
    error: ToolError | None = None
    text: str | None = None


def _short(exc: BaseException) -> str:
    return str(exc)[:500]


class Executor:
    def __init__(self, backends: dict[str, Backend], manifests: dict[str, Manifest],
                 embedders: EmbedderRegistry, *, hooks: Hooks | None = None,
                 policy: SourcePolicy | None = None, call_timeout: float = 30.0,
                 per_source_concurrency: int = 4, max_limit: int = 100, output_hits: int = 10,
                 snippet_chars: int = 240):
        self.backends = backends
        self.manifests = manifests
        self.embedders = embedders
        self.hooks = hooks or Hooks()
        self.policy = policy or SourcePolicy()
        self.call_timeout = call_timeout
        self.max_limit = max_limit
        self.output_hits = output_hits
        self.snippet_chars = snippet_chars
        self._sems = {name: asyncio.Semaphore(per_source_concurrency) for name in backends}

    # ---- validation ---------------------------------------------------------

    def validate(self, op: QueryOp) -> tuple[str, str] | None:
        manifest = self.manifests.get(op.source)
        if manifest is None or op.source not in self.backends:
            return "validation", f"unknown source {op.source!r}; available: {', '.join(sorted(self.manifests))}"
        missing = required_capabilities(op) - manifest.capabilities
        if missing:
            return "validation", (f"source {op.source!r} does not support "
                                  f"{', '.join(sorted(c.value for c in missing))}")
        coll = manifest.resolve_collection(op.collection)
        if coll is None and (op.collection is not None or op.type in _COLLECTION_SCOPED):
            names = ", ".join(c.name for c in manifest.collections)
            if op.collection is not None:
                return "validation", f"unknown collection {op.collection!r} in {op.source!r} (one of: {names})"
            return "validation", f"source {op.source!r} has several collections; pass collection (one of: {names})"
        if coll is None or not coll.fields:
            return None
        known = {f.name for f in coll.fields}
        referenced = filter_fields(op.filter) if op.filter is not None else set()
        if isinstance(op, (Lexical, Regex)) and op.fields:
            referenced |= set(op.fields)
        if isinstance(op, Aggregate):
            referenced |= set(op.group_by)
        if isinstance(op, (Vector, Hybrid)):
            referenced.add(op.field)
        unknown = referenced - known
        if unknown:
            return "validation", (f"unknown field(s) {sorted(unknown)} in collection {coll.name!r}; "
                                  f"known fields: {sorted(known)}")
        if isinstance(op, (Vector, Hybrid)):
            spec = coll.field(op.field)
            if spec is None or spec.type is not FieldType.VECTOR or not spec.embedder_id:
                return "validation", f"field {op.field!r} is not a vector field with a known embedder"
            embedder = self.embedders.get(spec.embedder_id)
            if embedder is None:
                return "embedder", (f"no embedder registered for {spec.embedder_id!r} (vector field "
                                    f"{op.field!r}); refusing to embed the query with a different model")
            part = op.query_content() if isinstance(op, Vector) else TextPart(text=op.text)
            if modality_of(part) not in embedder.modalities:
                return "embedder", f"embedder {embedder.id!r} cannot embed {modality_of(part).value} queries"
        return None

    # ---- execution ----------------------------------------------------------

    async def run(self, calls: list[ToolCall], *, question: Query, turn: int, pool: CandidatePool,
                  trace: Trace, model_id: str | None = None) -> ExecResult:
        outcomes = await asyncio.gather(*(self._run_one(c, question, turn, trace) for c in calls))
        result = ExecResult()
        for c, out in zip(calls, outcomes):
            if out.error is not None:
                result.errors.append(out.error)
                result.outputs[c.id] = out.error.render()
                trace.add("tool_error", turn, call_id=c.id, kind=out.error.kind,
                          message=out.error.message)
                continue
            if out.text is not None:
                result.outputs[c.id] = out.text
                continue
            new = pool.add(out.hits, turn=turn, call_id=c.id)
            result.new_keys.extend(new)
            result.hits_per_call[c.id] = [h.key for h in out.hits]
            result.outputs[c.id] = self._render_hits(out.hits, set(new), model_id)
            trace.add("tool_result", turn, call_id=c.id, n_hits=len(out.hits), n_new=len(new))
        return result

    async def _run_one(self, c: ToolCall, question: Query, turn: int, trace: Trace) -> _Outcome:
        trace.add("tool_call", turn, call_id=c.id, name=c.name, arguments=c.arguments)
        try:
            parsed = parse_call(c, question)
        except ValueError as exc:
            return _Outcome(error=ToolError(call_id=c.id, kind="validation", message=_short(exc)))
        if isinstance(parsed, DiscoverRequest):
            return self._discover(c.id, parsed)
        problem = self.validate(parsed)
        if problem is not None:
            kind, message = problem
            return _Outcome(error=ToolError(call_id=c.id, source=parsed.source, kind=kind,  # type: ignore[arg-type]
                                            message=message))
        op = parsed.model_copy(update={"limit": min(parsed.limit, self.max_limit)})
        if isinstance(op, (Vector, Hybrid)):
            try:
                op = await self._embed(op)
            except Exception as exc:
                return _Outcome(error=ToolError(call_id=c.id, source=op.source, kind="embedder",
                                                message=f"{type(exc).__name__}: {_short(exc)}"))
        try:
            async with self._sems[op.source]:
                hits = await asyncio.wait_for(self.backends[op.source].execute(op), self.call_timeout)
        except TimeoutError:
            return _Outcome(error=ToolError(call_id=c.id, source=op.source, kind="timeout",
                                            message=f"timed out after {self.call_timeout}s"))
        except Exception as exc:
            return _Outcome(error=ToolError(call_id=c.id, source=op.source, kind="backend",
                                            message=f"{type(exc).__name__}: {_short(exc)}"))
        return _Outcome(hits=hits)

    async def _embed(self, op: Vector | Hybrid) -> Vector | Hybrid:
        coll = self.manifests[op.source].resolve_collection(op.collection)
        spec = coll.field(op.field) if coll else None
        embedder = self.embedders.get(spec.embedder_id) if spec and spec.embedder_id else None
        assert embedder is not None  # guaranteed by validate()
        part = op.query_content() if isinstance(op, Vector) else TextPart(text=op.text)
        part = await self.hooks.before_model_call(embedder.id, part)
        [vector] = await embedder.embed([part], "query")
        return op.model_copy(update={"vector": list(vector)})

    def _discover(self, call_id: str, req: DiscoverRequest) -> _Outcome:
        manifest = self.manifests.get(req.source)
        if manifest is None:
            return _Outcome(error=ToolError(call_id=call_id, kind="validation",
                                            message=f"unknown source {req.source!r}"))
        if req.collection is None:
            return _Outcome(text=manifest.summary())
        coll = manifest.resolve_collection(req.collection)
        if coll is None:
            return _Outcome(error=ToolError(call_id=call_id, source=req.source, kind="validation",
                                            message=f"unknown collection {req.collection!r}"))
        return _Outcome(text=coll.model_dump_json(indent=1, exclude_none=True)[:MAX_DISCOVER_CHARS])

    def _render_hits(self, hits: list[Hit], new: set[str], model_id: str | None) -> str:
        lines = [f"{len(hits)} hits ({len(new)} new)"]
        for h in hits[: self.output_hits]:
            if model_id is None or self.policy.allows(h.source, model_id):
                body = h.snippet(self.snippet_chars)
            else:
                body = "(content withheld by source policy)"
            score = f" score={h.raw_score:.3f}" if h.raw_score is not None else ""
            lines.append(f"- {h.key}{score}: {body}")
        if len(hits) > self.output_hits:
            lines.append(f"... {len(hits) - self.output_hits} more")
        return "\n".join(lines)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/roles/test_executor.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(roles): executor with manifest validation, embedder binding, parallel execution

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Analyzer

**Files:**
- Create: `src/agentic_search/roles/analyzer.py`
- Test: `tests/roles/test_analyzer.py`

**Interfaces:**
- Consumes: `SearchState`, `Hooks`, `SourcePolicy` (Task 7), `ExecResult` (Task 9), and `Decider`, `ToolCall` (Task 6).
- Produces:
  - `AnalysisResult(n_new, n_new_relevant: int | None, digest: str, usage: ModelUsage)`.
  - `Analyzer(decider: Decider | None, *, hooks=None, policy=None, relevant_threshold=0.5, judge_weight=0.9, batch_size=16, timeout=60.0, digest_hits=8, snippet_chars=240)`.
  - `async .judge_keys(state, keys) -> ModelUsage`. It marks candidates judged or records `unjudged_reason`, and never raises.
  - `async .analyze(state, calls, result, *, digest_model_id) -> AnalysisResult`.
  - `.is_relevant(candidate) -> bool`.
  - Trace events: `judge` and `judge_error`.

- [ ] **Step 1: Write the failing tests**

`tests/roles/test_analyzer.py`:
```python
from agentic_search.core.hooks import SourcePolicy
from agentic_search.core.state import SearchState
from agentic_search.core.types import Budget, Hit, Query, TextPart
from agentic_search.roles.analyzer import Analyzer
from agentic_search.roles.executor import ExecResult
from agentic_search.testing import FailingDecider, KeywordJudge, ScriptedController, call


def setup_state():
    s = SearchState(question=Query.of("fever"), manifests={}, budget=Budget())
    hits = [Hit(doc_id="1", source="s", content=[TextPart(text="high fever")]),
            Hit(doc_id="2", source="s", content=[TextPart(text="castles")])]
    c = call("lexical_search", source="s", text="fever")
    new = s.pool.add(hits, turn=0, call_id=c.id)
    return s, [c], ExecResult(new_keys=new, hits_per_call={c.id: new})


async def test_judges_new_and_builds_digest():
    s, calls, res = setup_state()
    a = Analyzer(KeywordJudge(["fever"]))
    out = await a.analyze(s, calls, res, digest_model_id="driver")
    assert out.n_new == 2 and out.n_new_relevant == 1
    assert s.pool["s:1"].judged and s.pool["s:1"].p_relevant == 1.0
    assert "1 judged relevant" in out.digest
    assert "New relevant:" in out.digest and "s:1 p=1.00: high fever" in out.digest
    assert "Judged not relevant" in out.digest
    assert "2 hits, 2 new, 1 relevant" in out.digest
    assert len(s.trace.of_type("judge")) == 1


async def test_no_judge_leaves_unjudged():
    s, calls, res = setup_state()
    out = await Analyzer(None).analyze(s, calls, res, digest_model_id="driver")
    assert out.n_new_relevant is None
    assert s.pool["s:1"].unjudged_reason == "no judge configured"
    assert "New candidates (not judged)" in out.digest


async def test_failing_judge_fails_open():
    s, calls, res = setup_state()
    out = await Analyzer(FailingDecider()).analyze(s, calls, res, digest_model_id="d")
    assert out.n_new_relevant is None
    assert s.pool["s:1"].unjudged_reason.startswith("judge failed")
    assert s.trace.of_type("judge_error")[0].data["error"].startswith("RuntimeError")


async def test_decider_without_judge_support():
    s, calls, res = setup_state()
    a = Analyzer(ScriptedController([]))
    await a.analyze(s, calls, res, digest_model_id="d")
    assert "does not judge" in s.pool["s:1"].unjudged_reason


async def test_batching_and_policy():
    s, calls, res = setup_state()
    a = Analyzer(KeywordJudge(["fever"]), batch_size=1, policy=SourcePolicy({"s": {"judge-only"}}))
    out = await a.analyze(s, calls, res, digest_model_id="driver")
    assert len(s.trace.of_type("judge")) == 2
    # KeywordJudge is not allowed to see source s, so it sees empty content
    assert out.n_new_relevant == 0
    assert "withheld" in out.digest


async def test_error_lines_in_digest():
    from agentic_search.core.types import ToolError
    s, calls, res = setup_state()
    bad = call("lexical_search", source="x", text="y")
    res.errors.append(ToolError(call_id=bad.id, kind="validation", message="unknown source 'x'"))
    out = await Analyzer(None).analyze(s, [*calls, bad], res, digest_model_id="d")
    assert "ERROR [validation] unknown source 'x'" in out.digest
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/roles/test_analyzer.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`src/agentic_search/roles/analyzer.py`:
```python
"""Judges new candidates, blends scores, and writes the digest the planner reads next turn."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass

from agentic_search.core.hooks import Hooks, SourcePolicy
from agentic_search.core.state import Candidate, SearchState
from agentic_search.core.types import ModelUsage
from agentic_search.models.base import Decider, ToolCall
from agentic_search.roles.executor import ExecResult


@dataclass
class AnalysisResult:
    n_new: int
    n_new_relevant: int | None
    digest: str
    usage: ModelUsage


class Analyzer:
    def __init__(self, decider: Decider | None, *, hooks: Hooks | None = None,
                 policy: SourcePolicy | None = None, relevant_threshold: float = 0.5,
                 judge_weight: float = 0.9, batch_size: int = 16, timeout: float = 60.0,
                 digest_hits: int = 8, snippet_chars: int = 240):
        self.decider = decider
        self.hooks = hooks or Hooks()
        self.policy = policy or SourcePolicy()
        self.relevant_threshold = relevant_threshold
        self.judge_weight = judge_weight
        self.batch_size = batch_size
        self.timeout = timeout
        self.digest_hits = digest_hits
        self.snippet_chars = snippet_chars
        self._can_judge = decider is not None

    def is_relevant(self, cand: Candidate) -> bool:
        return cand.judged and cand.p_relevant is not None and cand.p_relevant >= self.relevant_threshold

    async def judge_keys(self, state: SearchState, keys: list[str]) -> ModelUsage:
        usage = ModelUsage()
        pending = [k for k in keys if not state.pool[k].judged]
        if not pending:
            return usage
        if not self._can_judge or self.decider is None:
            for k in pending:
                state.pool[k].unjudged_reason = "no judge configured"
            return usage
        decider = self.decider
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start:start + self.batch_size]
            hits = self.policy.redact([state.pool[k].hit for k in batch], decider.id)
            hits = await self.hooks.before_model_call(decider.id, hits)
            t0 = time.perf_counter()
            try:
                result = await asyncio.wait_for(decider.judge(state.question, hits), self.timeout)
            except NotImplementedError:
                self._can_judge = False
                for k in pending[start:]:
                    state.pool[k].unjudged_reason = f"{decider.id} does not judge"
                return usage
            except Exception as exc:
                for k in batch:
                    state.pool[k].unjudged_reason = f"judge failed: {type(exc).__name__}"
                state.trace.add("judge_error", state.turn, decider=decider.id, n=len(batch),
                                error=f"{type(exc).__name__}: {str(exc)[:300]}")
                continue
            usage = usage.plus(result.usage)
            by_key = {j.key: j for j in result.judgments}
            n_relevant = 0
            for k in batch:
                cand, j = state.pool[k], by_key.get(k)
                if j is None:
                    cand.unjudged_reason = "no judgment returned"
                    continue
                cand.p_relevant, cand.rationale = j.p_relevant, j.rationale
                cand.judged, cand.unjudged_reason = True, None
                n_relevant += self.is_relevant(cand)
            state.trace.add("judge", state.turn, duration_ms=(time.perf_counter() - t0) * 1000,
                            decider=decider.id, n=len(batch), n_relevant=n_relevant)
        return usage

    async def analyze(self, state: SearchState, calls: list[ToolCall], result: ExecResult, *,
                      digest_model_id: str) -> AnalysisResult:
        usage = await self.judge_keys(state, result.new_keys)
        judged_new = [state.pool[k] for k in result.new_keys if state.pool[k].judged]
        n_rel = sum(self.is_relevant(c) for c in judged_new) if judged_new else None
        digest = self._digest(state, calls, result, digest_model_id)
        return AnalysisResult(n_new=len(result.new_keys), n_new_relevant=n_rel, digest=digest,
                              usage=usage)

    def _digest(self, state: SearchState, calls: list[ToolCall], result: ExecResult,
                model_id: str) -> str:
        pool = state.pool
        new = [pool[k] for k in result.new_keys]
        relevant = [c for c in new if self.is_relevant(c)]
        irrelevant = [c for c in new if c.judged and not self.is_relevant(c)]
        header = f"Turn {state.turn}: {len(calls)} calls, {len(new)} new candidates"
        if any(c.judged for c in new):
            header += f", {len(relevant)} judged relevant"
        lines = [f"{header}. Pool size: {len(pool)}.", "Per call:"]
        errors = {e.call_id: e for e in result.errors}
        new_set = set(result.new_keys)
        for c in calls:
            args = json.dumps(c.arguments, default=str)[:200]
            if c.id in errors:
                lines.append(f"- {c.name} {args} -> {errors[c.id].render()}")
                continue
            keys = result.hits_per_call.get(c.id)
            if keys is None:
                lines.append(f"- {c.name} {args} -> ok")
                continue
            line = f"- {c.name} {args} -> {len(keys)} hits, {len(new_set.intersection(keys))} new"
            if self._can_judge:
                line += f", {sum(self.is_relevant(pool[k]) for k in keys)} relevant"
            lines.append(line)
        if relevant:
            lines.append("New relevant:")
            relevant.sort(key=lambda c: -(c.p_relevant or 0.0))
            lines.extend(self._line(c, model_id) for c in relevant[: self.digest_hits])
        if irrelevant:
            lines.append("Judged not relevant (examples):")
            lines.extend(self._line(c, model_id) for c in irrelevant[:3])
        if new and not relevant and not irrelevant:
            lines.append("New candidates (not judged):")
            lines.extend(self._line(c, model_id) for c in new[: self.digest_hits])
        top = pool.ranked(self.judge_weight)[:5]
        if top:
            lines.append("Current top results: " + ", ".join(f"{c.hit.key} ({s:.2f})" for c, s in top))
        return "\n".join(lines)

    def _line(self, cand: Candidate, model_id: str) -> str:
        h = cand.hit
        if self.policy.allows(h.source, model_id):
            body = h.snippet(self.snippet_chars)
        else:
            body = "(content withheld by source policy)"
        p = f" p={cand.p_relevant:.2f}" if cand.judged and cand.p_relevant is not None else ""
        why = f" ({cand.rationale})" if cand.rationale else ""
        return f"- {h.key}{p}: {body}{why}"
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/roles/test_analyzer.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(roles): analyzer with batched fail-open judging and planner digest

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Controller

**Files:**
- Create: `src/agentic_search/roles/controller.py`
- Test: `tests/roles/test_controller.py`

**Interfaces:**
- Consumes: `SearchState`, `Hooks` (Task 7), `Action`, `ControllerView`, `Decider`, `Decision` (Task 6), and `StopReason` (Task 2).
- Produces:
  - `Controller(decider: Decider | None, *, hooks=None, relevant_threshold=0.5, min_new_relevant=1, stable_top_k=10, timeout=60.0)`.
  - `.budget_stop(state) -> StopReason | None`, checked in the order turns, tool calls, tokens, cost, time.
  - `.budget_remaining(state) -> dict` and `.total_relevant(state) -> int | None`.
  - `.heuristic(state) -> Decision`.
  - `async .decide(state) -> Decision`. It never raises, and it traces a `decision` event (plus `decision_error` when the decider fails).

- [ ] **Step 1: Write the failing tests**

`tests/roles/test_controller.py`:
```python
from agentic_search.core.state import SearchState
from agentic_search.core.types import Budget, Hit, Query, StopReason
from agentic_search.models.base import Action, TurnSummary
from agentic_search.roles.controller import Controller
from agentic_search.testing import FailingDecider, KeywordJudge, ScriptedController


def state(budget=None, history=(), relevant=None):
    s = SearchState(question=Query.of("q"), manifests={}, budget=budget or Budget())
    s.history = list(history)
    if relevant is not None:
        s.pool.add([Hit(doc_id="a", source="s")], turn=0, call_id="c")
        s.pool["s:a"].judged, s.pool["s:a"].p_relevant = True, 1.0 if relevant else 0.0
    return s


def turn(n_calls=1, n_errors=0, n_new=2, n_new_relevant=1, top=("s:a",)):
    return TurnSummary(turn=0, n_calls=n_calls, n_errors=n_errors, n_new=n_new,
                       n_new_relevant=n_new_relevant, top_keys=list(top))


def test_budget_stop_order():
    c = Controller(None)
    s = state(Budget(max_turns=1, max_tool_calls=1))
    assert c.budget_stop(s) is None
    s.usage.turns = 1
    assert c.budget_stop(s) is StopReason.BUDGET_TURNS
    s = state(Budget(max_cost_usd=0.1, max_tokens=10, max_seconds=None))
    s.usage.input_tokens = 10
    assert c.budget_stop(s) is StopReason.BUDGET_TOKENS
    s.usage.input_tokens, s.usage.cost_usd = 0, 0.2
    assert c.budget_stop(s) is StopReason.BUDGET_COST
    s = state(Budget(max_seconds=0.0))
    assert c.budget_stop(s) is StopReason.BUDGET_TIME


def test_heuristic_rules():
    c = Controller(None)
    assert c.heuristic(state(history=[turn(n_calls=2, n_errors=2)])).action is Action.REFINE
    assert c.heuristic(state(history=[turn(n_new_relevant=0)], relevant=True)).action is Action.STOP
    assert c.heuristic(state(history=[turn(n_new_relevant=0)], relevant=False)).action is Action.BROADEN
    assert c.heuristic(state(history=[turn(n_new=0, n_new_relevant=None)])).action is Action.STOP
    assert c.heuristic(state(history=[turn(), turn()])).action is Action.STOP  # top stable
    assert c.heuristic(state(history=[turn(top=("x",)), turn()])).action is Action.CONTINUE


def test_budget_remaining():
    s = state(Budget(max_turns=3, max_tool_calls=10, max_cost_usd=1.0))
    s.usage.turns, s.usage.tool_calls = 1, 4
    r = Controller(None).budget_remaining(s)
    assert r["turns"] == 2 and r["tool_calls"] == 6 and r["cost_usd"] == 1.0 and r["tokens"] is None


async def test_decide_uses_decider_and_traces():
    ctrl = ScriptedController([Action.BROADEN])
    s = state(history=[turn()])
    d = await Controller(ctrl).decide(s)
    assert d.action is Action.BROADEN
    assert ctrl.views[0].history[0].n_new == 2
    assert s.trace.of_type("decision")[0].data["by"] == "scripted-controller"


async def test_decide_falls_back_to_heuristic():
    s = state(history=[turn(n_new=0, n_new_relevant=None)])
    d = await Controller(FailingDecider()).decide(s)
    assert d.action is Action.STOP
    assert s.trace.of_type("decision_error")
    s2 = state(history=[turn(n_new=0, n_new_relevant=None)])
    assert (await Controller(KeywordJudge([])).decide(s2)).action is Action.STOP
    assert s2.trace.of_type("decision")[0].data["by"] == "heuristic"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/roles/test_controller.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`src/agentic_search/roles/controller.py`:
```python
"""Decides whether to keep searching. Hard budgets always win over any decider."""

from __future__ import annotations

import asyncio
from typing import Any

from agentic_search.core.hooks import Hooks
from agentic_search.core.state import SearchState
from agentic_search.core.types import StopReason
from agentic_search.models.base import Action, ControllerView, Decider, Decision


class Controller:
    def __init__(self, decider: Decider | None, *, hooks: Hooks | None = None,
                 relevant_threshold: float = 0.5, min_new_relevant: int = 1,
                 stable_top_k: int = 10, timeout: float = 60.0):
        self.decider = decider
        self.hooks = hooks or Hooks()
        self.relevant_threshold = relevant_threshold
        self.min_new_relevant = min_new_relevant
        self.stable_top_k = stable_top_k
        self.timeout = timeout
        self._can_decide = decider is not None

    def budget_stop(self, state: SearchState) -> StopReason | None:
        b, u = state.budget, state.usage
        if u.turns >= b.max_turns:
            return StopReason.BUDGET_TURNS
        if u.tool_calls >= b.max_tool_calls:
            return StopReason.BUDGET_TOOL_CALLS
        if b.max_tokens is not None and u.total_tokens >= b.max_tokens:
            return StopReason.BUDGET_TOKENS
        if b.max_cost_usd is not None and u.cost_usd >= b.max_cost_usd:
            return StopReason.BUDGET_COST
        if b.max_seconds is not None and state.elapsed() >= b.max_seconds:
            return StopReason.BUDGET_TIME
        return None

    def budget_remaining(self, state: SearchState) -> dict[str, Any]:
        b, u = state.budget, state.usage
        return {
            "turns": b.max_turns - u.turns,
            "tool_calls": b.max_tool_calls - u.tool_calls,
            "tokens": None if b.max_tokens is None else b.max_tokens - u.total_tokens,
            "cost_usd": None if b.max_cost_usd is None else round(b.max_cost_usd - u.cost_usd, 4),
            "seconds": None if b.max_seconds is None else round(b.max_seconds - state.elapsed(), 1),
        }

    def total_relevant(self, state: SearchState) -> int | None:
        judged = [c for c in state.pool.candidates() if c.judged and c.p_relevant is not None]
        if not judged:
            return None
        return sum(c.p_relevant >= self.relevant_threshold for c in judged)  # type: ignore[operator]

    def heuristic(self, state: SearchState) -> Decision:
        if not state.history:
            return Decision(action=Action.CONTINUE, note="no turns yet")
        last = state.history[-1]
        if last.n_calls and last.n_errors == last.n_calls:
            return Decision(action=Action.REFINE, note="every call failed; fix the errors")
        if last.n_new_relevant is not None:
            if last.n_new_relevant < self.min_new_relevant:
                if (self.total_relevant(state) or 0) > 0:
                    return Decision(action=Action.STOP, note="no new relevant results this turn")
                return Decision(action=Action.BROADEN, note="nothing relevant yet; broaden or rephrase")
        elif last.n_new == 0:
            return Decision(action=Action.STOP, note="no new candidates")
        if len(state.history) >= 2:
            k = self.stable_top_k
            prev, cur = state.history[-2].top_keys[:k], last.top_keys[:k]
            if cur and cur == prev:
                return Decision(action=Action.STOP, note="top results stable")
        return Decision(action=Action.CONTINUE, note="new results found; keep exploring")

    async def decide(self, state: SearchState) -> Decision:
        by = "heuristic"
        if self.decider is None or not self._can_decide:
            decision = self.heuristic(state)
        else:
            decider = self.decider
            view = ControllerView(question=state.question, turn=state.turn, history=state.history,
                                  digest=state.digest, total_relevant=self.total_relevant(state),
                                  budget_remaining=self.budget_remaining(state))
            view = await self.hooks.before_model_call(decider.id, view)
            try:
                decision = await asyncio.wait_for(decider.decide(view), self.timeout)
                by = decider.id
            except NotImplementedError:
                self._can_decide = False
                decision = self.heuristic(state)
            except Exception as exc:
                state.trace.add("decision_error", state.turn, decider=decider.id,
                                error=f"{type(exc).__name__}: {str(exc)[:300]}")
                decision = self.heuristic(state)
        state.trace.add("decision", state.turn, action=decision.action.value, note=decision.note,
                        confidence=decision.confidence, by=by)
        return decision
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/roles/test_controller.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(roles): controller with budget enforcement, heuristic, and decider fallback

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---
### Task 12: Planner + annotations

**Files:**
- Create: `src/agentic_search/roles/planner.py`, `src/agentic_search/core/annotations.py`
- Test: `tests/roles/test_planner.py`, `tests/core/test_annotations.py`

**Interfaces:**
- Consumes: `Driver`, `PlannerView`, `PlanResult`, `ToolSpec` (Task 6), and `SearchState`, `Hooks` (Task 7).
- Produces:
  - `Planner(driver, *, hooks=None, max_calls_per_turn=8)`. `async .plan(state, tools) -> PlanResult` truncates to `max_calls_per_turn`, renames call ids to `t{turn}.c{i}`, and traces a `plan` event.
  - `render_manifests(manifests) -> str`.
  - `apply_annotations(manifest, ann: dict | None) -> Manifest`. It returns a deep copy. The annotation schema is `{description, collections: {name: {description, fields: {field: {description, type, embedder_id, vector_dim, vector_metric, searchable, filterable, sortable}}}}}`.

- [ ] **Step 1: Write the failing tests**

`tests/roles/test_planner.py`:
```python
from agentic_search.core.hooks import Hooks
from agentic_search.core.state import SearchState
from agentic_search.core.types import Budget, Capability, Manifest, Query
from agentic_search.models.base import Action, Decision
from agentic_search.roles.planner import Planner, render_manifests
from agentic_search.testing import ScriptedDriver, call


def state():
    m = Manifest(source="docs", backend_type="files", capabilities={Capability.LEXICAL})
    s = SearchState(question=Query.of("q"), manifests={"docs": m}, budget=Budget())
    s.turn, s.digest = 2, "digest text"
    s.last_decision = Decision(action=Action.BROADEN, note="wider")
    return s


async def test_plan_truncates_renames_and_traces():
    d = ScriptedDriver([[call("a"), call("b"), call("c")]])
    s = state()
    res = await Planner(d, max_calls_per_turn=2).plan(s, [])
    assert [c.id for c in res.calls] == ["t2.c0", "t2.c1"]
    v = d.views[0]
    assert v.digest == "digest text" and v.directive.action is Action.BROADEN and v.max_calls == 2
    assert "`docs`" in v.manifest_summary
    ev = s.trace.of_type("plan")[0]
    assert ev.data["n_calls"] == 2 and ev.data["dropped"] == 1


async def test_hooks_can_rewrite_view():
    class Redact(Hooks):
        async def before_model_call(self, model_id, payload):
            return payload.model_copy(update={"digest": "[redacted]"})

    d = ScriptedDriver([[]])
    await Planner(d, hooks=Redact()).plan(state(), [])
    assert d.views[0].digest == "[redacted]"


def test_render_manifests_sorted():
    a = Manifest(source="a", backend_type="x", capabilities=set())
    b = Manifest(source="b", backend_type="x", capabilities=set())
    out = render_manifests({"b": b, "a": a})
    assert out.index("`a`") < out.index("`b`")
```

`tests/core/test_annotations.py`:
```python
from agentic_search.core.annotations import apply_annotations
from agentic_search.core.types import CollectionInfo, FieldSpec, FieldType, Manifest


def base():
    return Manifest(source="s", backend_type="x", capabilities=set(), collections=[
        CollectionInfo(name="notes", fields=[FieldSpec(name="dx_code", type=FieldType.KEYWORD)])])


def test_apply_annotations():
    m = base()
    out = apply_annotations(m, {
        "description": "Clinical notes",
        "collections": {"notes": {
            "description": "one row per note",
            "fields": {
                "dx_code": {"description": "ICD-10-CM code", "filterable": True},
                "emb": {"type": "vector", "embedder_id": "azure:medsiglip-448", "vector_dim": 1152},
            }},
            "ghost": {"description": "ignored"}},
    })
    coll = out.resolve_collection("notes")
    assert out.description == "Clinical notes" and coll.description == "one row per note"
    assert coll.field("dx_code").description == "ICD-10-CM code" and coll.field("dx_code").filterable
    emb = coll.field("emb")
    assert emb.type is FieldType.VECTOR and emb.embedder_id == "azure:medsiglip-448"
    assert m.description is None and m.resolve_collection("notes").field("emb") is None


def test_none_is_identity():
    m = base()
    assert apply_annotations(m, None) is m
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/roles/test_planner.py tests/core/test_annotations.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement planner**

`src/agentic_search/roles/planner.py`:
```python
"""Builds the planner's view of the search and asks the driver for the next batch of calls."""

from __future__ import annotations

import time

from agentic_search.core.hooks import Hooks
from agentic_search.core.state import SearchState
from agentic_search.core.types import Manifest
from agentic_search.models.base import Driver, PlannerView, PlanResult, ToolSpec


def render_manifests(manifests: dict[str, Manifest]) -> str:
    return "\n\n".join(m.summary() for _, m in sorted(manifests.items()))


class Planner:
    def __init__(self, driver: Driver, *, hooks: Hooks | None = None, max_calls_per_turn: int = 8):
        self.driver = driver
        self.hooks = hooks or Hooks()
        self.max_calls = max_calls_per_turn

    async def plan(self, state: SearchState, tools: list[ToolSpec]) -> PlanResult:
        view = PlannerView(question=state.question, turn=state.turn,
                           manifest_summary=render_manifests(state.manifests), digest=state.digest,
                           directive=state.last_decision, errors=state.last_errors,
                           max_calls=self.max_calls)
        view = await self.hooks.before_model_call(self.driver.id, view)
        t0 = time.perf_counter()
        result = await self.driver.plan(view, tools)
        calls = [c.model_copy(update={"id": f"t{state.turn}.c{i}"})
                 for i, c in enumerate(result.calls[: self.max_calls])]
        state.trace.add("plan", state.turn, duration_ms=(time.perf_counter() - t0) * 1000,
                        driver=self.driver.id, n_calls=len(calls),
                        dropped=max(0, len(result.calls) - self.max_calls), note=result.note)
        return result.model_copy(update={"calls": calls})
```

- [ ] **Step 4: Implement annotations**

`src/agentic_search/core/annotations.py`:
```python
"""Merge human-written meaning into discovered manifests (e.g. 'dx_code is ICD-10-CM')."""

from __future__ import annotations

from typing import Any

from agentic_search.core.types import FieldSpec, FieldType, Manifest

_FIELD_KEYS = ("description", "embedder_id", "vector_dim", "vector_metric",
               "searchable", "filterable", "sortable")


def apply_annotations(manifest: Manifest, ann: dict[str, Any] | None) -> Manifest:
    """Return a copy of `manifest` with annotations applied. Unknown collections are ignored;
    unknown fields are added (useful to declare vector fields introspection can't see)."""
    if not ann:
        return manifest
    out = manifest.model_copy(deep=True)
    if "description" in ann:
        out.description = ann["description"]
    for cname, cann in (ann.get("collections") or {}).items():
        coll = out.resolve_collection(cname)
        if coll is None:
            continue
        if "description" in cann:
            coll.description = cann["description"]
        for fname, fann in (cann.get("fields") or {}).items():
            spec = coll.field(fname)
            if spec is None:
                spec = FieldSpec(name=fname, type=FieldType(fann.get("type", "keyword")))
                coll.fields.append(spec)
            elif "type" in fann:
                spec.type = FieldType(fann["type"])
            for key in _FIELD_KEYS:
                if key in fann:
                    setattr(spec, key, fann[key])
    return out
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/roles/test_planner.py tests/core/test_annotations.py -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: planner role and manifest annotations

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: Harness (loop, modes, delegate runtime, finalize)

**Files:**
- Create: `src/agentic_search/core/harness.py`
- Modify: `src/agentic_search/__init__.py` (exports)
- Test: `tests/core/test_harness.py`

**Interfaces:**
- Consumes: every role (Tasks 8–12), `apply_annotations` and `render_manifests` (Task 12).
- Produces:
  - `Mode = Literal["retrieval","harness","model"]` and `HarnessError`.
  - `HarnessSettings(max_calls_per_turn=8, call_timeout=30.0, per_source_concurrency=4, max_limit=100, relevant_threshold=0.5, judge_weight=0.9, judge_batch_size=16, decider_timeout=60.0, min_new_relevant=1)`.
  - `RankedHit(hit, score, p_relevant, rationale, judged)`.
  - `SearchResult(question, hits, stop_reason, usage, trace, mode)` with `.keys()`.
  - `Harness(backends, driver, *, embedders=(), analyzer=None, controller=None, mode="harness", budget=None, hooks=None, source_policy=None, annotations=None, settings=None)`, with:
    - `async .setup() -> dict[str, Manifest]`, plus `.setup_errors`
    - `async .search(question: str | Query, *, sources=None, top_k=20, mode=None, budget=None) -> SearchResult`
    - `async .close()` and async context manager support.
- Package exports: `from agentic_search import Harness, HarnessSettings, SearchResult, Query, Budget`.

- [ ] **Step 1: Write the failing tests**

`tests/core/test_harness.py`:
```python
import pytest

from agentic_search import Budget, Harness, Query
from agentic_search.backends.files import FilesBackend
from agentic_search.core.harness import HarnessError
from agentic_search.core.hooks import Hooks
from agentic_search.core.types import StopReason
from agentic_search.models.base import Action
from agentic_search.testing import (
    FailingDecider, KeywordJudge, ScriptedController, ScriptedDriver, call,
)


def lex(text, **kw):
    return call("lexical_search", source="docs", text=text, **kw)


def make(docs_backend, driver, **kw):
    return Harness([docs_backend], driver, embedders=[docs_backend.embedder], **kw)


async def test_harness_mode_loop(docs_backend):
    driver = ScriptedDriver([[lex("headache")], [lex("fever")]])
    h = make(docs_backend, driver, analyzer=KeywordJudge(["headache", "fever"]))
    res = await h.search("what treats headache?")
    assert set(res.keys()) == {"docs:d1", "docs:d4"}
    assert res.stop_reason is StopReason.CONTROLLER_STOP  # turn 1 found nothing new
    assert res.usage.turns == 2 and res.usage.tool_calls == 2
    assert all(r.judged and r.p_relevant == 1.0 for r in res.hits)
    turn1 = driver.views[1]
    assert "docs:d1" in turn1.digest and turn1.directive.action is Action.CONTINUE
    assert driver.views[0].digest.startswith("No searches")


async def test_retrieval_mode_single_pass(docs_backend):
    driver = ScriptedDriver([[lex("headache")], [lex("fever")]])
    res = await make(docs_backend, driver).search("q", mode="retrieval")
    assert res.stop_reason is StopReason.SINGLE_PASS and len(driver.views) == 1
    assert set(res.keys()) == {"docs:d1", "docs:d4"} and not res.hits[0].judged


async def test_budget_turns_and_tool_calls(docs_backend):
    judge = KeywordJudge(["headache"])
    d1 = ScriptedDriver([[lex("headache")], [lex("pain")]])
    r1 = await make(docs_backend, d1, analyzer=judge).search("q", budget=Budget(max_turns=1))
    assert r1.stop_reason is StopReason.BUDGET_TURNS
    d2 = ScriptedDriver([[lex("headache"), lex("pain")]])
    r2 = await make(docs_backend, d2, analyzer=judge).search("q", budget=Budget(max_tool_calls=1))
    assert r2.stop_reason is StopReason.BUDGET_TOOL_CALLS
    assert len(r2.trace.of_type("tool_call")) == 1


async def test_no_plan(docs_backend):
    res = await make(docs_backend, ScriptedDriver([])).search("q")
    assert res.stop_reason is StopReason.NO_PLAN and res.hits == []


async def test_controller_decider(docs_backend):
    driver = ScriptedDriver([[lex("headache")], [lex("pain")], [lex("castles")]])
    ctrl = ScriptedController([Action.CONTINUE, Action.STOP])
    res = await make(docs_backend, driver, controller=ctrl).search("q")
    assert res.stop_reason is StopReason.CONTROLLER_STOP and len(driver.views) == 2


async def test_model_mode_delegate(docs_backend):
    driver = ScriptedDriver(delegate_calls=[[lex("headache", id="a")]],
                            delegate_keys=["docs:d4", "docs:d1", "docs:nope"])
    res = await make(docs_backend, driver).search("q", mode="model")
    assert res.keys() == ["docs:d4", "docs:d1"]
    assert [h.score for h in res.hits] == [1.0, 0.5]
    assert res.stop_reason is StopReason.DELEGATE_DONE
    assert "docs:d1" in driver.delegate_outputs[0][0]
    assert "`docs`" in driver.delegate_context
    assert res.trace.of_type("delegate")[0].data["unknown_keys"] == ["docs:nope"]


async def test_model_mode_rerank_with_judge(docs_backend):
    driver = ScriptedDriver(delegate_calls=[[lex("headache", id="a")]],
                            delegate_keys=["docs:d4", "docs:d1"])
    res = await make(docs_backend, driver, analyzer=KeywordJudge(["aspirin"])).search("q", mode="model")
    assert res.keys() == ["docs:d1", "docs:d4"]


async def test_model_mode_budget(docs_backend):
    driver = ScriptedDriver(delegate_calls=[[lex("headache", id="a")], [lex("pain", id="b")]])
    res = await make(docs_backend, driver).search("q", mode="model", budget=Budget(max_tool_calls=1))
    assert driver.delegate_outputs[1][0].startswith("[budget]")
    assert res.stop_reason is StopReason.BUDGET_TOOL_CALLS


async def test_failing_judge_fails_open(docs_backend):
    driver = ScriptedDriver([[lex("headache")]])
    res = await make(docs_backend, driver, analyzer=FailingDecider()).search("q")
    assert res.hits and not any(h.judged for h in res.hits)
    assert res.trace.of_type("judge_error")
    assert res.stop_reason is StopReason.NO_PLAN


async def test_hooks_see_every_model_call(docs_backend):
    class Recording(Hooks):
        def __init__(self):
            self.models, self.events = [], []

        async def before_model_call(self, model_id, payload):
            self.models.append(model_id)
            return payload

        def on_trace_event(self, event):
            self.events.append(event.type)

    hooks = Recording()
    driver = ScriptedDriver([[lex("headache"), call("vector_search", source="docs",
                                                    field="embedding", hyde_text="fever")]])
    await make(docs_backend, driver, analyzer=KeywordJudge(["x"]), hooks=hooks).search("q")
    assert {"scripted-driver", "keyword-judge", "hash"} <= set(hooks.models)
    assert {"setup", "plan", "tool_call", "finalize"} <= set(hooks.events)


class BrokenBackend:
    name, backend_type = "broken", "x"

    def capabilities(self):
        return set()

    async def discover(self, detail="full", collection=None):
        raise ConnectionError("no route")

    async def execute(self, op):
        return []

    async def close(self):
        pass


async def test_setup_errors(docs_backend):
    h = Harness([docs_backend, BrokenBackend()], ScriptedDriver([]))
    await h.setup()
    assert "broken" in h.setup_errors and set(h.manifests) == {"docs"}
    with pytest.raises(HarnessError):
        await Harness([BrokenBackend()], ScriptedDriver([])).search("q")


async def test_sources_and_config_errors(docs_backend, medical_docs):
    h = make(docs_backend, ScriptedDriver([]))
    with pytest.raises(HarnessError):
        await h.search("q", sources=["nope"])
    with pytest.raises(HarnessError):
        await h.search("q", mode="bogus")
    with pytest.raises(HarnessError):
        Harness([docs_backend, FilesBackend.from_documents("docs", medical_docs)], ScriptedDriver([]))
    with pytest.raises(HarnessError):
        Harness([docs_backend], ScriptedDriver([]), annotations={"nope": {}})


async def test_annotations_applied_and_result_serializes(docs_backend):
    h = make(docs_backend, ScriptedDriver([[lex("headache")]]),
             annotations={"docs": {"description": "Drug notes"}})
    res = await h.search(Query.of("q"))
    assert h.manifests["docs"].description == "Drug notes"
    assert '"stop_reason"' in res.model_dump_json()
    async with h:
        pass
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/core/test_harness.py -q`
Expected: FAIL with `ImportError: cannot import name 'Harness'`

- [ ] **Step 3: Implement**

`src/agentic_search/core/harness.py`:
```python
"""The Harness: Plan → Execute → Analyze → Decide over any backends, with swappable models."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Literal, Sequence

from pydantic import BaseModel

from agentic_search.backends.base import Backend
from agentic_search.core.annotations import apply_annotations
from agentic_search.core.hooks import Hooks, SourcePolicy
from agentic_search.core.state import Candidate, SearchState, Trace, Usage
from agentic_search.core.types import Budget, Hit, Manifest, Query, StopReason
from agentic_search.embedders.base import Embedder, EmbedderRegistry
from agentic_search.models.base import Action, Decider, Driver, ToolCall, ToolSpec, TurnSummary
from agentic_search.roles.analyzer import Analyzer
from agentic_search.roles.controller import Controller
from agentic_search.roles.executor import Executor
from agentic_search.roles.planner import Planner, render_manifests
from agentic_search.roles.tools import build_tool_specs

Mode = Literal["retrieval", "harness", "model"]
_MODES = ("retrieval", "harness", "model")


class HarnessError(Exception):
    """Harness-level failure: misconfiguration or no reachable backends."""


class HarnessSettings(BaseModel):
    max_calls_per_turn: int = 8
    call_timeout: float = 30.0
    per_source_concurrency: int = 4
    max_limit: int = 100
    relevant_threshold: float = 0.5
    judge_weight: float = 0.9
    judge_batch_size: int = 16
    decider_timeout: float = 60.0
    min_new_relevant: int = 1


class RankedHit(BaseModel):
    hit: Hit
    score: float
    p_relevant: float | None = None
    rationale: str | None = None
    judged: bool = False


class SearchResult(BaseModel):
    question: Query
    hits: list[RankedHit]
    stop_reason: StopReason
    usage: Usage
    trace: Trace
    mode: str

    def keys(self) -> list[str]:
        return [h.hit.key for h in self.hits]


class _DelegateRuntime:
    """ToolRuntime for model-centric drivers: executes calls, enforces budgets, records the trace."""

    def __init__(self, state: SearchState, executor: Executor, controller: Controller, model_id: str):
        self.state, self.executor, self.controller, self.model_id = state, executor, controller, model_id
        self.exhausted: StopReason | None = None

    async def call(self, calls: list[ToolCall]) -> list[str]:
        state = self.state
        reason = self.controller.budget_stop(state)
        allowed: list[ToolCall] = []
        if reason is None:
            allowed = calls[: max(0, state.budget.max_tool_calls - state.usage.tool_calls)]
            if len(allowed) < len(calls):
                reason = StopReason.BUDGET_TOOL_CALLS
        if reason is not None and self.exhausted is None:
            self.exhausted = reason
        message = (f"[budget] {reason.value}: stop searching and call finish with your ranked keys."
                   if reason is not None else "")
        outputs: dict[str, str] = {}
        if allowed:
            result = await self.executor.run(allowed, question=state.question, turn=state.turn,
                                             pool=state.pool, trace=state.trace,
                                             model_id=self.model_id)
            state.usage.tool_calls += len(allowed)
            state.usage.turns += 1
            state.turn += 1
            outputs = result.outputs
        return [outputs.get(c.id, message) for c in calls]


class Harness:
    def __init__(self, backends: Sequence[Backend], driver: Driver, *,
                 embedders: Sequence[Embedder] = (), analyzer: Decider | None = None,
                 controller: Decider | None = None, mode: Mode = "harness",
                 budget: Budget | None = None, hooks: Hooks | None = None,
                 source_policy: dict[str, set[str]] | None = None,
                 annotations: dict[str, dict[str, Any]] | None = None,
                 settings: HarnessSettings | None = None):
        if not backends:
            raise HarnessError("at least one backend is required")
        names = [b.name for b in backends]
        if len(set(names)) != len(names):
            raise HarnessError(f"duplicate backend names: {names}")
        if mode not in _MODES:
            raise HarnessError(f"unknown mode {mode!r}; one of {_MODES}")
        self.backends = {b.name: b for b in backends}
        self.annotations = annotations or {}
        unknown = set(self.annotations) - set(self.backends)
        if unknown:
            raise HarnessError(f"annotations for unknown sources: {sorted(unknown)}")
        self.driver = driver
        self.embedders = EmbedderRegistry(embedders)
        self.analyzer_decider = analyzer
        self.controller_decider = controller
        self.mode: Mode = mode
        self.budget = budget or Budget()
        self.hooks = hooks or Hooks()
        self.policy = SourcePolicy(source_policy)
        self.settings = settings or HarnessSettings()
        self.manifests: dict[str, Manifest] = {}
        self.setup_errors: dict[str, str] = {}
        self._ready = False
        self._setup_lock = asyncio.Lock()

    async def setup(self) -> dict[str, Manifest]:
        async with self._setup_lock:
            if self._ready:
                return self.manifests
            names = list(self.backends)
            results = await asyncio.gather(*(self.backends[n].discover("full") for n in names),
                                           return_exceptions=True)
            for name, res in zip(names, results):
                if isinstance(res, BaseException):
                    self.setup_errors[name] = f"{type(res).__name__}: {res}"
                    continue
                self.manifests[name] = apply_annotations(res, self.annotations.get(name))
            if not self.manifests:
                raise HarnessError(f"no backend could be discovered: {self.setup_errors}")
            self._ready = True
            return self.manifests

    async def search(self, question: str | Query, *, sources: list[str] | None = None,
                     top_k: int = 20, mode: Mode | None = None,
                     budget: Budget | None = None) -> SearchResult:
        run_mode = mode or self.mode
        if run_mode not in _MODES:
            raise HarnessError(f"unknown mode {run_mode!r}; one of {_MODES}")
        await self.setup()
        query = Query.of(question) if isinstance(question, str) else question
        manifests = self._select(sources)
        state = SearchState(question=query, manifests=manifests, budget=budget or self.budget)
        state.trace.set_listener(self.hooks.on_trace_event)
        state.trace.add("setup", 0, sources=sorted(manifests), setup_errors=self.setup_errors,
                        mode=run_mode)
        s = self.settings
        executor = Executor({n: self.backends[n] for n in manifests}, manifests, self.embedders,
                            hooks=self.hooks, policy=self.policy, call_timeout=s.call_timeout,
                            per_source_concurrency=s.per_source_concurrency, max_limit=s.max_limit)
        analyzer = Analyzer(self.analyzer_decider, hooks=self.hooks, policy=self.policy,
                            relevant_threshold=s.relevant_threshold, judge_weight=s.judge_weight,
                            batch_size=s.judge_batch_size, timeout=s.decider_timeout)
        controller = Controller(self.controller_decider, hooks=self.hooks,
                                relevant_threshold=s.relevant_threshold,
                                min_new_relevant=s.min_new_relevant, timeout=s.decider_timeout)
        tools = build_tool_specs(manifests)
        order: list[str] | None = None
        if run_mode == "model":
            reason, order = await self._run_delegate(state, executor, analyzer, controller, tools)
        else:
            reason = await self._run_loop(state, executor, analyzer, controller, tools,
                                          single_pass=run_mode == "retrieval")
        return self._finalize(state, reason, top_k, order, run_mode)

    def _select(self, sources: list[str] | None) -> dict[str, Manifest]:
        if sources is None:
            return dict(self.manifests)
        unknown = [s for s in sources if s not in self.manifests]
        if unknown:
            raise HarnessError(f"unknown or undiscovered sources: {unknown}")
        return {s: self.manifests[s] for s in sources}

    async def _run_loop(self, state: SearchState, executor: Executor, analyzer: Analyzer,
                        controller: Controller, tools: list[ToolSpec], *,
                        single_pass: bool) -> StopReason:
        planner = Planner(self.driver, hooks=self.hooks,
                          max_calls_per_turn=self.settings.max_calls_per_turn)
        while True:
            reason = controller.budget_stop(state)
            if reason is not None:
                state.trace.add("budget", state.turn, reason=reason.value)
                return reason
            plan = await planner.plan(state, tools)
            state.usage.add_model(plan.usage)
            calls = plan.calls[: state.budget.max_tool_calls - state.usage.tool_calls]
            if not calls:
                return StopReason.NO_PLAN
            result = await executor.run(calls, question=state.question, turn=state.turn,
                                        pool=state.pool, trace=state.trace, model_id=self.driver.id)
            state.usage.tool_calls += len(calls)
            state.usage.turns += 1
            analysis = await analyzer.analyze(state, calls, result, digest_model_id=self.driver.id)
            state.usage.add_model(analysis.usage)
            state.digest, state.last_errors = analysis.digest, result.errors
            top = [c.hit.key for c, _ in state.pool.ranked(self.settings.judge_weight)[:20]]
            state.history.append(TurnSummary(
                turn=state.turn, n_calls=len(calls), n_errors=len(result.errors),
                n_new=analysis.n_new, n_new_relevant=analysis.n_new_relevant, top_keys=top))
            if single_pass:
                return StopReason.SINGLE_PASS
            decision = await controller.decide(state)
            state.usage.add_model(decision.usage)
            if decision.action is Action.STOP:
                return StopReason.CONTROLLER_STOP
            state.last_decision = decision
            state.turn += 1

    async def _run_delegate(self, state: SearchState, executor: Executor, analyzer: Analyzer,
                            controller: Controller,
                            tools: list[ToolSpec]) -> tuple[StopReason, list[str] | None]:
        runtime = _DelegateRuntime(state, executor, controller, self.driver.id)
        t0 = time.perf_counter()
        result = await self.driver.run_delegate(state.question, tools, runtime, state.budget,
                                                context=render_manifests(state.manifests))
        state.usage.add_model(result.usage)
        ranked = [k for k in dict.fromkeys(result.ranked_keys) if k in state.pool]
        unknown = [k for k in result.ranked_keys if k not in state.pool]
        state.trace.add("delegate", state.turn, duration_ms=(time.perf_counter() - t0) * 1000,
                        driver=self.driver.id, n_ranked=len(ranked), unknown_keys=unknown[:20],
                        note=result.note)
        if self.analyzer_decider is not None:
            keys = ranked or [c.hit.key for c in state.pool.candidates()]
            state.usage.add_model(await analyzer.judge_keys(state, keys))
        return (runtime.exhausted or StopReason.DELEGATE_DONE), (ranked or None)

    def _finalize(self, state: SearchState, reason: StopReason, top_k: int,
                  order: list[str] | None, mode: str) -> SearchResult:
        w = self.settings.judge_weight
        ranked: list[tuple[Candidate, float]]
        if order:
            cands = [state.pool[k] for k in order]
            if any(c.judged for c in cands):
                scores = state.pool.scores(w)
                ranked = sorted(((c, scores[c.hit.key]) for c in cands),
                                key=lambda cs: (-cs[1], cs[0].hit.key))
            else:
                ranked = [(c, 1.0 / (i + 1)) for i, c in enumerate(cands)]
        else:
            ranked = state.pool.ranked(w)
        hits = [RankedHit(hit=c.hit, score=round(s, 6), p_relevant=c.p_relevant,
                          rationale=c.rationale, judged=c.judged) for c, s in ranked[:top_k]]
        state.trace.add("finalize", state.turn, stop_reason=reason.value, n_hits=len(hits),
                        pool_size=len(state.pool))
        return SearchResult(question=state.question, hits=hits, stop_reason=reason,
                            usage=state.usage, trace=state.trace, mode=mode)

    async def close(self) -> None:
        await asyncio.gather(*(b.close() for b in self.backends.values()), return_exceptions=True)

    async def __aenter__(self) -> Harness:
        await self.setup()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
```

Replace `src/agentic_search/__init__.py`:
```python
"""Agentic search harness."""

from agentic_search.core.harness import Harness, HarnessError, HarnessSettings, RankedHit, SearchResult
from agentic_search.core.types import Budget, Query

__version__ = "0.1.0"

__all__ = ["Budget", "Harness", "HarnessError", "HarnessSettings", "Query", "RankedHit",
           "SearchResult", "__version__"]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest -q`
Expected: all pass, including every earlier test

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(core): harness loop with retrieval/harness/model modes, budgets, finalize

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---
### Task 14: LLM chat layer + Anthropic client

**Files:**
- Create: `src/agentic_search/models/llm.py`, `src/agentic_search/models/anthropic.py`
- Modify: `src/agentic_search/testing.py` (append `FakeLLMClient`)
- Test: `tests/models/test_anthropic.py`

**Interfaces:**
- Consumes: `Content`, `ImagePart`, `image_bytes`, `text_of`, `ModelUsage` (Task 2), and `ToolCall`, `ToolSpec` (Task 6).
- Produces:
  - `ChatMessage(role: "user"|"assistant"|"tool", content: str | list[Content] = "", tool_calls=[], tool_call_id=None)`.
  - `ChatResponse(text, tool_calls, usage, stop_reason)`.
  - The `LLMClient` protocol: `id`, `supports_images`, `async chat(system, messages, *, tools=None, tool_choice=None, max_tokens=4096) -> ChatResponse`. Here `tool_choice` is a tool name to force.
  - `usage_with_cost(input_tokens, output_tokens, price_per_mtok) -> ModelUsage`.
  - `AnthropicClient(model, *, client=None, api_key=None, supports_images=True, price_per_mtok=None, id=None, max_retries=3)`. Its id defaults to `anthropic:{model}`.
  - `testing.FakeLLMClient(responses, *, id="fake-llm", supports_images=False)` records `.requests`.

- [ ] **Step 1: Write the failing tests**

`tests/models/test_anthropic.py`:
```python
from types import SimpleNamespace

import pytest

from agentic_search.core.types import ImagePart, TextPart
from agentic_search.models.anthropic import AnthropicClient
from agentic_search.models.base import ToolCall, ToolSpec
from agentic_search.models.llm import ChatMessage, ChatResponse, usage_with_cost
from agentic_search.testing import FakeLLMClient


class FakeMessages:
    def __init__(self, response):
        self.response, self.kwargs = response, None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def sdk(blocks, stop="tool_use", tin=1000, tout=200):
    resp = SimpleNamespace(content=blocks, stop_reason=stop,
                           usage=SimpleNamespace(input_tokens=tin, output_tokens=tout))
    return SimpleNamespace(messages=FakeMessages(resp))


TOOL = ToolSpec(name="lexical_search", description="d", parameters={"type": "object"})


def test_usage_with_cost():
    u = usage_with_cost(1_000_000, 500_000, (3.0, 15.0))
    assert u.cost_usd == pytest.approx(10.5)
    assert usage_with_cost(5, 5, None).cost_usd == 0.0


async def test_parses_text_and_tool_use():
    fake = sdk([SimpleNamespace(type="text", text="thinking"),
                SimpleNamespace(type="tool_use", id="tu1", name="lexical_search",
                                input={"source": "s", "text": "x"})])
    c = AnthropicClient("claude-sonnet-5-5", client=fake, price_per_mtok=(3.0, 15.0))
    assert c.id == "anthropic:claude-sonnet-5-5"
    r = await c.chat("sys", [ChatMessage(role="user", content="hi")], tools=[TOOL],
                     tool_choice="lexical_search")
    assert r.text == "thinking"
    assert r.tool_calls == [ToolCall(id="tu1", name="lexical_search",
                                     arguments={"source": "s", "text": "x"})]
    assert r.usage.input_tokens == 1000 and r.usage.cost_usd == pytest.approx(0.006)
    kw = fake.messages.kwargs
    assert kw["system"] == "sys" and kw["model"] == "claude-sonnet-5-5"
    assert kw["tools"] == [{"name": "lexical_search", "description": "d",
                            "input_schema": {"type": "object"}}]
    assert kw["tool_choice"] == {"type": "tool", "name": "lexical_search"}


async def test_message_conversion_merges_tool_results():
    fake = sdk([SimpleNamespace(type="text", text="ok")], stop="end_turn")
    c = AnthropicClient("m", client=fake)
    calls = [ToolCall(id="a", name="t", arguments={}), ToolCall(id="b", name="t", arguments={})]
    await c.chat("s", [
        ChatMessage(role="user", content="q"),
        ChatMessage(role="assistant", content="", tool_calls=calls),
        ChatMessage(role="tool", tool_call_id="a", content="r1"),
        ChatMessage(role="tool", tool_call_id="b", content="r2"),
    ])
    msgs = fake.messages.kwargs["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"] == [
        {"type": "tool_use", "id": "a", "name": "t", "input": {}},
        {"type": "tool_use", "id": "b", "name": "t", "input": {}},
    ]
    assert [b["tool_use_id"] for b in msgs[2]["content"]] == ["a", "b"]


async def test_images():
    fake = sdk([SimpleNamespace(type="text", text="")], stop="end_turn")
    content = [TextPart(text="look"), ImagePart(data=b"\x89PNG", mime="image/png")]
    await AnthropicClient("m", client=fake).chat("s", [ChatMessage(role="user", content=content)])
    blocks = fake.messages.kwargs["messages"][0]["content"]
    assert blocks[1]["type"] == "image" and blocks[1]["source"]["type"] == "base64"
    await AnthropicClient("m", client=fake, supports_images=False).chat(
        "s", [ChatMessage(role="user", content=content)])
    assert fake.messages.kwargs["messages"][0]["content"][1] == {"type": "text", "text": "[image omitted]"}


async def test_fake_llm_client():
    f = FakeLLMClient([ChatResponse(text="a")])
    assert (await f.chat("s", [ChatMessage(role="user", content="x")])).text == "a"
    assert f.requests[0]["system"] == "s"
    with pytest.raises(AssertionError):
        await f.chat("s", [])
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/models/test_anthropic.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the chat layer**

`src/agentic_search/models/llm.py`:
```python
"""Provider-neutral chat interface that drivers and LLM judges are built on."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agentic_search.core.types import Content, ModelUsage
from agentic_search.models.base import ToolCall, ToolSpec


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str | list[Content] = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None


class ChatResponse(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: ModelUsage = Field(default_factory=ModelUsage)
    stop_reason: str | None = None


@runtime_checkable
class LLMClient(Protocol):
    id: str
    supports_images: bool

    async def chat(self, system: str, messages: list[ChatMessage], *,
                   tools: list[ToolSpec] | None = None, tool_choice: str | None = None,
                   max_tokens: int = 4096) -> ChatResponse: ...


def usage_with_cost(input_tokens: int, output_tokens: int,
                    price_per_mtok: tuple[float, float] | None) -> ModelUsage:
    cost = 0.0
    if price_per_mtok is not None:
        cost = (input_tokens * price_per_mtok[0] + output_tokens * price_per_mtok[1]) / 1_000_000
    return ModelUsage(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost)
```

- [ ] **Step 4: Implement the Anthropic client**

`src/agentic_search/models/anthropic.py`:
```python
"""Anthropic Messages API adapter. Needs the `anthropic` extra unless a client is injected."""

from __future__ import annotations

import base64
from typing import Any

from agentic_search.core.types import Content, ImagePart, image_bytes, text_of
from agentic_search.models.base import ToolCall, ToolSpec
from agentic_search.models.llm import ChatMessage, ChatResponse, usage_with_cost


def _as_text(content: str | list[Content]) -> str:
    return content if isinstance(content, str) else text_of(content)


def _image_block(part: ImagePart) -> dict[str, Any]:
    if part.uri is not None and part.uri.startswith(("http://", "https://")):
        return {"type": "image", "source": {"type": "url", "url": part.uri}}
    data = base64.b64encode(image_bytes(part)).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": part.mime, "data": data}}


class AnthropicClient:
    def __init__(self, model: str, *, client: Any = None, api_key: str | None = None,
                 supports_images: bool = True, price_per_mtok: tuple[float, float] | None = None,
                 id: str | None = None, max_retries: int = 3):
        if client is None:
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic(api_key=api_key, max_retries=max_retries)
        self.model = model
        self._client = client
        self.supports_images = supports_images
        self.price = price_per_mtok
        self.id = id or f"anthropic:{model}"

    async def chat(self, system: str, messages: list[ChatMessage], *,
                   tools: list[ToolSpec] | None = None, tool_choice: str | None = None,
                   max_tokens: int = 4096) -> ChatResponse:
        kwargs: dict[str, Any] = {"model": self.model, "max_tokens": max_tokens, "system": system,
                                  "messages": self._messages(messages)}
        if tools:
            kwargs["tools"] = [{"name": t.name, "description": t.description,
                                "input_schema": t.parameters} for t in tools]
        if tool_choice:
            kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}
        resp = await self._client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(id=b.id, name=b.name, arguments=dict(b.input or {}))
                 for b in resp.content if b.type == "tool_use"]
        return ChatResponse(text=text, tool_calls=calls, stop_reason=resp.stop_reason,
                            usage=usage_with_cost(resp.usage.input_tokens,
                                                  resp.usage.output_tokens, self.price))

    def _messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "tool":
                block = {"type": "tool_result", "tool_use_id": m.tool_call_id,
                         "content": _as_text(m.content)}
                prev = out[-1] if out else None
                if (prev and prev["role"] == "user" and isinstance(prev["content"], list)
                        and all(b.get("type") == "tool_result" for b in prev["content"])):
                    prev["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            elif m.role == "assistant":
                blocks: list[dict[str, Any]] = []
                if _as_text(m.content):
                    blocks.append({"type": "text", "text": _as_text(m.content)})
                blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                           for c in m.tool_calls]
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": "user", "content": self._user_content(m.content)})
        return out

    def _user_content(self, content: str | list[Content]) -> str | list[dict[str, Any]]:
        if isinstance(content, str):
            return content
        blocks: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, ImagePart):
                blocks.append(_image_block(part) if self.supports_images
                              else {"type": "text", "text": "[image omitted]"})
            else:
                blocks.append({"type": "text", "text": text_of([part])})
        return blocks
```

- [ ] **Step 5: Append FakeLLMClient to testing.py**

Append to `src/agentic_search/testing.py`:
```python


class FakeLLMClient:
    """LLMClient that replays scripted ChatResponses and records every request."""

    def __init__(self, responses: list[Any], *, id: str = "fake-llm", supports_images: bool = False):
        self.responses = list(responses)
        self.id = id
        self.supports_images = supports_images
        self.requests: list[dict[str, Any]] = []

    async def chat(self, system: str, messages: list[Any], *, tools: list[ToolSpec] | None = None,
                   tool_choice: str | None = None, max_tokens: int = 4096) -> Any:
        self.requests.append({"system": system,
                              "messages": [m.model_copy(deep=True) for m in messages],
                              "tools": tools, "tool_choice": tool_choice})
        if not self.responses:
            raise AssertionError("FakeLLMClient ran out of scripted responses")
        return self.responses.pop(0)
```

- [ ] **Step 6: Run to verify pass**

Run: `uv run pytest tests/models -q`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(models): provider-neutral chat layer and Anthropic client

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 15: OpenAI-compatible client

**Files:**
- Create: `src/agentic_search/models/openai_compat.py`
- Test: `tests/models/test_openai_compat.py`

**Interfaces:**
- Consumes: Task 14.
- Produces: `OpenAICompatClient(model, *, base_url=None, api_key=None, client=None, supports_images=False, price_per_mtok=None, id=None, max_tokens_param="max_tokens", max_retries=3)`. Its id defaults to `openai:{model}`. Tool arguments that aren't valid JSON become `{"_invalid_json": raw}`, which `parse_call` rejects.

- [ ] **Step 1: Write the failing tests**

`tests/models/test_openai_compat.py`:
```python
from types import SimpleNamespace

from agentic_search.core.types import ImagePart, TextPart
from agentic_search.models.base import ToolCall, ToolSpec
from agentic_search.models.llm import ChatMessage
from agentic_search.models.openai_compat import OpenAICompatClient


class FakeCompletions:
    def __init__(self, response):
        self.response, self.kwargs = response, None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def sdk(content="hi", tool_calls=None, usage=(10, 5)):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")],
                           usage=SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1]))
    return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(resp)))


def tc(id, name, args):
    return SimpleNamespace(id=id, function=SimpleNamespace(name=name, arguments=args))


async def test_parses_tool_calls_and_usage():
    fake = sdk(tool_calls=[tc("a", "lexical_search", '{"source":"s","text":"x"}'),
                           tc("b", "lexical_search", "{oops")])
    c = OpenAICompatClient("qwen3", client=fake, price_per_mtok=(1.0, 2.0))
    assert c.id == "openai:qwen3"
    r = await c.chat("sys", [ChatMessage(role="user", content="q")],
                     tools=[ToolSpec(name="lexical_search", description="d", parameters={})],
                     tool_choice="lexical_search", max_tokens=100)
    assert r.text == "hi" and r.tool_calls[0].arguments == {"source": "s", "text": "x"}
    assert r.tool_calls[1].arguments == {"_invalid_json": "{oops"}
    assert r.usage.input_tokens == 10 and r.usage.cost_usd == (10 * 1 + 5 * 2) / 1e6
    kw = fake.chat.completions.kwargs
    assert kw["messages"][0] == {"role": "system", "content": "sys"}
    assert kw["tools"][0]["function"]["name"] == "lexical_search"
    assert kw["tool_choice"] == {"type": "function", "function": {"name": "lexical_search"}}
    assert kw["max_tokens"] == 100


async def test_message_conversion():
    fake = sdk(content=None)
    c = OpenAICompatClient("m", client=fake, supports_images=True, max_tokens_param="max_completion_tokens")
    await c.chat("s", [
        ChatMessage(role="user", content=[TextPart(text="look"), ImagePart(data=b"x", mime="image/png")]),
        ChatMessage(role="assistant", content="", tool_calls=[ToolCall(id="a", name="t", arguments={"k": 1})]),
        ChatMessage(role="tool", tool_call_id="a", content="result"),
    ])
    kw = fake.chat.completions.kwargs
    user, asst, tool = kw["messages"][1:]
    assert user["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert asst == {"role": "assistant", "content": None, "tool_calls": [
        {"id": "a", "type": "function", "function": {"name": "t", "arguments": '{"k": 1}'}}]}
    assert tool == {"role": "tool", "tool_call_id": "a", "content": "result"}
    assert "max_completion_tokens" in kw


async def test_missing_usage_and_content():
    fake = sdk(content=None, tool_calls=None)
    fake.chat.completions.response.usage = None
    r = await OpenAICompatClient("m", client=fake).chat("s", [ChatMessage(role="user", content="q")])
    assert r.text == "" and r.tool_calls == [] and r.usage.input_tokens == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/models/test_openai_compat.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`src/agentic_search/models/openai_compat.py`:
```python
"""OpenAI-compatible chat completions adapter: OpenAI, vLLM, Ollama, Together, Baseten
(e.g. SID-1), and most open-model servers. Needs the `openai` extra unless a client is injected."""

from __future__ import annotations

import base64
import json
import os
from typing import Any

from agentic_search.core.types import Content, ImagePart, image_bytes, text_of
from agentic_search.models.base import ToolCall, ToolSpec
from agentic_search.models.llm import ChatMessage, ChatResponse, usage_with_cost


def _as_text(content: str | list[Content]) -> str:
    return content if isinstance(content, str) else text_of(content)


def _parse_args(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"_invalid_json": raw}
    return value if isinstance(value, dict) else {"_invalid_json": raw}


class OpenAICompatClient:
    def __init__(self, model: str, *, base_url: str | None = None, api_key: str | None = None,
                 client: Any = None, supports_images: bool = False,
                 price_per_mtok: tuple[float, float] | None = None, id: str | None = None,
                 max_tokens_param: str = "max_tokens", max_retries: int = 3):
        if client is None:
            from openai import AsyncOpenAI
            key = api_key or os.environ.get("OPENAI_API_KEY") or "unused"
            client = AsyncOpenAI(base_url=base_url, api_key=key, max_retries=max_retries)
        self.model = model
        self._client = client
        self.supports_images = supports_images
        self.price = price_per_mtok
        self.max_tokens_param = max_tokens_param
        self.id = id or f"openai:{model}"

    async def chat(self, system: str, messages: list[ChatMessage], *,
                   tools: list[ToolSpec] | None = None, tool_choice: str | None = None,
                   max_tokens: int = 4096) -> ChatResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *self._messages(messages)],
            self.max_tokens_param: max_tokens,
        }
        if tools:
            kwargs["tools"] = [{"type": "function", "function": {
                "name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools]
        if tool_choice:
            kwargs["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}
        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
        calls = [ToolCall(id=tc.id, name=tc.function.name, arguments=_parse_args(tc.function.arguments))
                 for tc in (msg.tool_calls or [])]
        u = resp.usage
        usage = usage_with_cost(getattr(u, "prompt_tokens", 0) or 0,
                                getattr(u, "completion_tokens", 0) or 0, self.price)
        return ChatResponse(text=msg.content or "", tool_calls=calls, usage=usage,
                            stop_reason=choice.finish_reason)

    def _messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "tool":
                out.append({"role": "tool", "tool_call_id": m.tool_call_id,
                            "content": _as_text(m.content)})
            elif m.role == "assistant":
                entry: dict[str, Any] = {"role": "assistant", "content": _as_text(m.content) or None}
                if m.tool_calls:
                    entry["tool_calls"] = [{"id": c.id, "type": "function", "function": {
                        "name": c.name, "arguments": json.dumps(c.arguments)}} for c in m.tool_calls]
                out.append(entry)
            else:
                out.append({"role": "user", "content": self._user_content(m.content)})
        return out

    def _user_content(self, content: str | list[Content]) -> str | list[dict[str, Any]]:
        if isinstance(content, str):
            return content
        parts: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, ImagePart):
                if not self.supports_images:
                    parts.append({"type": "text", "text": "[image omitted]"})
                    continue
                if part.uri is not None and part.uri.startswith(("http://", "https://")):
                    url = part.uri
                else:
                    url = f"data:{part.mime};base64,{base64.b64encode(image_bytes(part)).decode()}"
                parts.append({"type": "image_url", "image_url": {"url": url}})
            else:
                parts.append({"type": "text", "text": text_of([part])})
        return parts
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/models/test_openai_compat.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(models): OpenAI-compatible chat client (vLLM, Ollama, Together, Baseten)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 16: ToolCallingDriver

**Files:**
- Create: `src/agentic_search/models/driver.py`
- Test: `tests/models/test_driver.py`

**Interfaces:**
- Consumes: `LLMClient`, `ChatMessage` (Task 14), and `Driver`, `PlannerView`, `PlanResult`, `DelegateResult`, `ToolRuntime`, `ToolSpec` (Task 6).
- Produces:
  - `ToolCallingDriver(client: LLMClient, *, max_calls_per_turn=8, planner_prompt=None, delegate_prompt=None, max_tokens=4096)`. Its `id` and `supports_images` come from the client.
  - `FINISH_TOOL: ToolSpec` (`finish(ranked_keys: list[str])`).
  - `render_planner_view(view) -> str`.

- [ ] **Step 1: Write the failing tests**

`tests/models/test_driver.py`:
```python
from agentic_search.core.types import Budget, ImagePart, Query, TextPart
from agentic_search.models.base import Action, Decision, Driver, PlannerView, ToolCall, ToolSpec
from agentic_search.models.driver import FINISH_TOOL, ToolCallingDriver, render_planner_view
from agentic_search.models.llm import ChatResponse
from agentic_search.testing import FakeLLMClient

TOOLS = [ToolSpec(name="lexical_search", description="d", parameters={})]


def tc(i, name="lexical_search", **args):
    return ToolCall(id=f"id{i}", name=name, arguments=args)


def view(question=None):
    return PlannerView(question=question or Query.of("what treats headache?"), turn=1,
                       manifest_summary="## source `docs`", digest="Turn 0: stuff",
                       directive=Decision(action=Action.REFINE, note="be precise"), max_calls=2)


def test_render_planner_view():
    text = render_planner_view(view())
    for s in ("what treats headache?", "## source `docs`", "# Turn 1", "Turn 0: stuff",
              "REFINE", "be precise", "up to 2 tool calls"):
        assert s in text


async def test_plan_truncates_and_passes_tools():
    client = FakeLLMClient([ChatResponse(text="plan", tool_calls=[tc(1), tc(2), tc(3)])])
    d = ToolCallingDriver(client, max_calls_per_turn=8)
    assert isinstance(d, Driver) and d.id == "fake-llm"
    res = await d.plan(view(), TOOLS)
    assert len(res.calls) == 2 and res.note == "plan"
    req = client.requests[0]
    assert req["tools"] == TOOLS and "up to 2" in req["system"]
    assert isinstance(req["messages"][0].content, str)


async def test_plan_includes_question_images_when_supported():
    q = Query(content=[TextPart(text="similar scans?"), ImagePart(data=b"x")])
    client = FakeLLMClient([ChatResponse()], supports_images=True)
    await ToolCallingDriver(client).plan(view(q), TOOLS)
    content = client.requests[0]["messages"][0].content
    assert isinstance(content, list) and isinstance(content[1], ImagePart)


class Runtime:
    def __init__(self):
        self.batches = []

    async def call(self, calls):
        self.batches.append(calls)
        return [f"result for {c.id}" for c in calls]


async def test_delegate_loop_until_finish():
    client = FakeLLMClient([
        ChatResponse(tool_calls=[tc(1, source="docs", text="headache")]),
        ChatResponse(text="done", tool_calls=[tc(2, name="finish", ranked_keys=["docs:d1"])]),
    ])
    rt = Runtime()
    res = await ToolCallingDriver(client).run_delegate(Query.of("q"), TOOLS, rt, Budget(),
                                                        context="## source `docs`")
    assert res.ranked_keys == ["docs:d1"] and res.note == "done"
    assert len(rt.batches) == 1
    second = client.requests[1]["messages"]
    assert [m.role for m in second] == ["user", "assistant", "tool"]
    assert second[2].content == "result for id1"
    assert "## source `docs`" in second[0].content
    assert client.requests[0]["tools"][-1] == FINISH_TOOL


async def test_delegate_forces_finish_after_budget_turns():
    client = FakeLLMClient([
        ChatResponse(tool_calls=[tc(1)]),
        ChatResponse(tool_calls=[tc(2)]),
        ChatResponse(tool_calls=[tc(3, name="finish", ranked_keys=["a", 7])]),
    ])
    res = await ToolCallingDriver(client).run_delegate(Query.of("q"), TOOLS, Runtime(),
                                                        Budget(max_turns=1))
    assert res.ranked_keys == ["a", "7"]
    assert client.requests[2]["tool_choice"] == "finish"
    assert client.requests[2]["tools"] == [FINISH_TOOL]


async def test_delegate_stops_when_model_stops_calling_tools():
    client = FakeLLMClient([ChatResponse(text="nothing to do")])
    res = await ToolCallingDriver(client).run_delegate(Query.of("q"), TOOLS, Runtime(), Budget())
    assert res.ranked_keys == [] and res.note == "nothing to do"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/models/test_driver.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`src/agentic_search/models/driver.py`:
```python
"""A Driver built on any LLMClient with tool calling (Claude, GPT, open models, SID-1-style)."""

from __future__ import annotations

from typing import Any

from agentic_search.core.types import Budget, Content, ModelUsage, Query, TextPart
from agentic_search.models.base import (
    DelegateResult, PlannerView, PlanResult, ToolCall, ToolRuntime, ToolSpec,
)
from agentic_search.models.llm import ChatMessage, LLMClient

PLANNER_PROMPT = """You are the planning component of an agentic search system. Your job is to find \
every document in the available data sources that is relevant to the user's question. You do not \
answer the question.

Each turn you see the question, a description of each data source (collections, fields, \
capabilities), a digest of what earlier searches returned and which results were judged relevant, \
and sometimes a directive from the controller (REFINE, BROADEN, SWITCH_SOURCE, CONTINUE).

Respond with tool calls only. Guidelines:
- Issue several diverse searches in parallel each turn (up to {max_calls}): narrow and broad keyword \
searches, synonyms and domain terminology, and vector searches whose hyde_text is a short passage \
written the way a relevant document would be written.
- Use filters when the question implies constraints and the source has matching filterable fields.
- Use only sources, collections and fields that appear in the source descriptions. Call discover \
for more detail on a collection.
- Never repeat a search that already ran. Learn from what was judged relevant and not relevant.
- If a call failed, read the error and correct it.
- If you believe the search is complete, make no tool calls."""

DELEGATE_PROMPT = """You are a search agent. Find every document in the data sources below that is \
relevant to the user's question. You do not answer the question.

Search iteratively: issue several diverse searches in parallel per turn (keyword variants, synonyms, \
vector searches with a hypothetical relevant passage as hyde_text, filtered searches), read the \
results, and refine. Results list documents as `source:doc_id` keys. Use only the sources, \
collections and fields described below. When you have found enough, call finish with ranked_keys: \
the keys of the relevant documents, most relevant first. If told the budget is exhausted, call \
finish immediately.

# Data sources
{context}"""

FINISH_TOOL = ToolSpec(
    name="finish",
    description="End the search and return the relevant document keys, most relevant first.",
    parameters={"type": "object",
                "properties": {"ranked_keys": {"type": "array", "items": {"type": "string"}}},
                "required": ["ranked_keys"]},
)


def render_planner_view(view: PlannerView) -> str:
    parts = [f"# Question\n{view.question.as_text()}"]
    images = view.question.images()
    if images:
        parts.append(f"(The question includes {len(images)} image(s).)")
    parts.append(f"# Data sources\n{view.manifest_summary}")
    parts.append(f"# Turn {view.turn}\n## Results so far\n{view.digest}")
    if view.directive is not None:
        note = f": {view.directive.note}" if view.directive.note else ""
        parts.append(f"## Controller directive: {view.directive.action.value.upper()}{note}")
    parts.append(f"Issue up to {view.max_calls} tool calls now, or none if the search is complete.")
    return "\n\n".join(parts)


def _finish_keys(call: ToolCall) -> list[str]:
    keys = call.arguments.get("ranked_keys", [])
    return [str(k) for k in keys] if isinstance(keys, list) else []


class ToolCallingDriver:
    def __init__(self, client: LLMClient, *, max_calls_per_turn: int = 8,
                 planner_prompt: str | None = None, delegate_prompt: str | None = None,
                 max_tokens: int = 4096):
        self.client = client
        self.id = client.id
        self.supports_images = client.supports_images
        self.max_calls = max_calls_per_turn
        self.planner_prompt = planner_prompt or PLANNER_PROMPT
        self.delegate_prompt = delegate_prompt or DELEGATE_PROMPT
        self.max_tokens = max_tokens

    def _with_images(self, text: str, question: Query) -> str | list[Content]:
        images = question.images() if self.supports_images else []
        return [TextPart(text=text), *images] if images else text

    async def plan(self, view: PlannerView, tools: list[ToolSpec]) -> PlanResult:
        limit = min(view.max_calls, self.max_calls)
        system = self.planner_prompt.format(max_calls=limit)
        content = self._with_images(render_planner_view(view), view.question)
        resp = await self.client.chat(system, [ChatMessage(role="user", content=content)],
                                      tools=tools, max_tokens=self.max_tokens)
        return PlanResult(calls=resp.tool_calls[:limit], usage=resp.usage, note=resp.text or None)

    async def run_delegate(self, question: Query, tools: list[ToolSpec], runtime: ToolRuntime,
                           budget: Budget, context: str = "") -> DelegateResult:
        system = self.delegate_prompt.format(context=context or "(see tool descriptions)")
        opening = f"{context}\n\n# Question\n{question.as_text()}" if context else question.as_text()
        messages = [ChatMessage(role="user", content=self._with_images(opening, question))]
        usage = ModelUsage()
        all_tools = [*tools, FINISH_TOOL]
        for _ in range(budget.max_turns + 1):
            resp = await self.client.chat(system, messages, tools=all_tools, max_tokens=self.max_tokens)
            usage = usage.plus(resp.usage)
            finish = next((c for c in resp.tool_calls if c.name == FINISH_TOOL.name), None)
            if finish is not None:
                return DelegateResult(ranked_keys=_finish_keys(finish), usage=usage,
                                      note=resp.text or None)
            if not resp.tool_calls:
                return DelegateResult(ranked_keys=[], usage=usage, note=resp.text or None)
            calls = resp.tool_calls[: self.max_calls]
            messages.append(ChatMessage(role="assistant", content=resp.text, tool_calls=calls))
            outputs = await runtime.call(calls)
            messages.extend(ChatMessage(role="tool", tool_call_id=c.id, content=o)
                            for c, o in zip(calls, outputs))
        resp = await self.client.chat(system, messages, tools=[FINISH_TOOL],
                                      tool_choice=FINISH_TOOL.name, max_tokens=self.max_tokens)
        usage = usage.plus(resp.usage)
        finish = next((c for c in resp.tool_calls if c.name == FINISH_TOOL.name), None)
        keys: list[Any] = _finish_keys(finish) if finish is not None else []
        return DelegateResult(ranked_keys=keys, usage=usage, note=resp.text or None)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/models/test_driver.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(models): tool-calling driver with planner and delegate (model-centric) loops

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---
### Task 17: LLM judge + cross-encoder judge

**Files:**
- Create: `src/agentic_search/models/llm_judge.py`, `src/agentic_search/models/cross_encoder.py`
- Test: `tests/models/test_judges.py`

**Interfaces:**
- Consumes: `LLMClient`, `ChatMessage` (Task 14), and the `Decider` data types (Task 6).
- Produces:
  - `LLMJudge(client, *, snippet_chars=1200, id=None, max_tokens=2048)`. Its id defaults to `llm-judge:{client.id}`. It implements both `judge` and `decide` through forced tool calls, and raises `ValueError` if the model skips the tool (the Analyzer and Controller then fail open).
  - `CrossEncoderJudge(model_name="BAAI/bge-reranker-v2-m3", *, scorer=None, snippet_chars=2000, apply_sigmoid=False, id=None)`. It implements `judge` only. Its id defaults to `cross-encoder:{model_name}`.

- [ ] **Step 1: Write the failing tests**

`tests/models/test_judges.py`:
```python
import pytest

from agentic_search import Harness
from agentic_search.core.types import Hit, Query, TextPart
from agentic_search.models.base import Action, ControllerView, ToolCall, TurnSummary
from agentic_search.models.cross_encoder import CrossEncoderJudge
from agentic_search.models.driver import ToolCallingDriver
from agentic_search.models.llm import ChatResponse
from agentic_search.models.llm_judge import LLMJudge
from agentic_search.testing import FakeLLMClient

HITS = [Hit(doc_id="1", source="s", content=[TextPart(text="aspirin for headache")]),
        Hit(doc_id="2", source="s", content=[TextPart(text="castles")])]


def submit(name, **args):
    return ChatResponse(tool_calls=[ToolCall(id="x", name=name, arguments=args)])


async def test_llm_judge_maps_indices_and_clamps():
    client = FakeLLMClient([submit("submit_judgments", judgments=[
        {"index": 0, "p_relevant": 1.4, "rationale": "on topic"},
        {"index": 1, "p_relevant": 0.1},
        {"index": 9, "p_relevant": 0.5},
        {"index": "bad"},
    ])])
    j = LLMJudge(client)
    assert j.id == "llm-judge:fake-llm"
    res = await j.judge(Query.of("headache"), HITS)
    assert [(x.key, x.p_relevant) for x in res.judgments] == [("s:1", 1.0), ("s:2", 0.1)]
    req = client.requests[0]
    assert req["tool_choice"] == "submit_judgments"
    assert "[0] s:1" in req["messages"][0].content[0].text


async def test_llm_judge_requires_tool_call():
    with pytest.raises(ValueError):
        await LLMJudge(FakeLLMClient([ChatResponse(text="no")])).judge(Query.of("q"), HITS)
    assert (await LLMJudge(FakeLLMClient([])).judge(Query.of("q"), [])).judgments == []


async def test_llm_judge_decide():
    client = FakeLLMClient([submit("submit_decision", action="broaden", confidence=0.7, note="n")])
    view = ControllerView(question=Query.of("q"), turn=1, digest="d", total_relevant=0,
                          history=[TurnSummary(turn=0, n_calls=2, n_errors=0, n_new=3,
                                               n_new_relevant=0)],
                          budget_remaining={"turns": 2})
    d = await LLMJudge(client).decide(view)
    assert d.action is Action.BROADEN and d.confidence == 0.7
    assert "turn 0: 2 calls" in client.requests[0]["messages"][0].content


async def test_cross_encoder_with_injected_scorer():
    seen = []

    def scorer(pairs):
        seen.extend(pairs)
        return [2.0, -2.0]

    j = CrossEncoderJudge("m", scorer=scorer, apply_sigmoid=True)
    res = await j.judge(Query.of("headache"), HITS)
    assert seen[0] == ("headache", "aspirin for headache")
    assert res.judgments[0].p_relevant == pytest.approx(0.8808, abs=1e-3)
    assert res.judgments[1].p_relevant == pytest.approx(0.1192, abs=1e-3)
    with pytest.raises(NotImplementedError):
        await j.decide(None)
    clamp = CrossEncoderJudge("m", scorer=lambda p: [1.7, -0.2])
    assert [x.p_relevant for x in (await clamp.judge(Query.of("q"), HITS)).judgments] == [1.0, 0.0]


async def test_end_to_end_with_llm_driver_and_judge(docs_backend):
    driver_client = FakeLLMClient([
        ChatResponse(tool_calls=[ToolCall(id="1", name="lexical_search",
                                          arguments={"source": "docs", "text": "headache"})]),
        ChatResponse(text="done"),
    ])
    judge_client = FakeLLMClient([submit("submit_judgments", judgments=[
        {"index": 0, "p_relevant": 0.9}, {"index": 1, "p_relevant": 0.8}])], id="judge")
    h = Harness([docs_backend], ToolCallingDriver(driver_client), analyzer=LLMJudge(judge_client))
    res = await h.search("what treats headache?")
    assert set(res.keys()) == {"docs:d1", "docs:d4"} and all(r.judged for r in res.hits)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/models/test_judges.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the LLM judge**

`src/agentic_search/models/llm_judge.py`:
```python
"""Decider backed by any LLMClient: relevance grading and stop/continue decisions."""

from __future__ import annotations

import json

from agentic_search.core.types import Content, Hit, ImagePart, Query, TextPart
from agentic_search.models.base import (
    Action, ControllerView, Decision, JudgeResult, Judgment, ToolSpec,
)
from agentic_search.models.llm import ChatMessage, LLMClient

JUDGE_SYSTEM = """You grade search results for relevance to a question. For each numbered document, \
estimate the probability (0 to 1) that it is relevant, meaning it contains information that helps \
answer the question. Be calibrated: 0.9+ only for clearly relevant documents, under 0.2 for \
off-topic ones. Give a rationale of at most 20 words. Call submit_judgments with one entry per \
document."""

DECIDE_SYSTEM = """You control an agentic search loop. Given the question, per-turn statistics, the \
remaining budget and a digest of the latest results, choose the next action: STOP if enough relevant \
documents were found or more searching is unlikely to help; REFINE to make queries more precise; \
BROADEN to widen them; SWITCH_SOURCE to try other sources; CONTINUE otherwise. Call submit_decision."""

JUDGE_TOOL = ToolSpec(name="submit_judgments", description="Submit one judgment per document.",
                      parameters={"type": "object", "properties": {"judgments": {
                          "type": "array", "items": {"type": "object", "properties": {
                              "index": {"type": "integer"},
                              "p_relevant": {"type": "number", "minimum": 0, "maximum": 1},
                              "rationale": {"type": "string"}},
                              "required": ["index", "p_relevant"]}}},
                          "required": ["judgments"]})

DECISION_TOOL = ToolSpec(name="submit_decision", description="Submit the next action.",
                         parameters={"type": "object", "properties": {
                             "action": {"type": "string", "enum": [a.value for a in Action]},
                             "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                             "note": {"type": "string"}},
                             "required": ["action"]})


def _clamp(x: float) -> float:
    return min(1.0, max(0.0, x))


class LLMJudge:
    def __init__(self, client: LLMClient, *, snippet_chars: int = 1200, id: str | None = None,
                 max_tokens: int = 2048):
        self.client = client
        self.snippet_chars = snippet_chars
        self.max_tokens = max_tokens
        self.id = id or f"llm-judge:{client.id}"

    async def judge(self, question: Query, hits: list[Hit]) -> JudgeResult:
        if not hits:
            return JudgeResult(judgments=[])
        docs = "\n\n".join(f"[{i}] {h.key}\n{h.snippet(self.snippet_chars) or '(no content)'}"
                           for i, h in enumerate(hits))
        content: list[Content] = [TextPart(text=f"Question: {question.as_text()}\n\nDocuments:\n{docs}")]
        if self.client.supports_images:
            content += question.images()
            for i, h in enumerate(hits):
                for part in h.content:
                    if isinstance(part, ImagePart):
                        content += [TextPart(text=f"Image for document [{i}]:"), part]
        resp = await self.client.chat(JUDGE_SYSTEM, [ChatMessage(role="user", content=content)],
                                      tools=[JUDGE_TOOL], tool_choice=JUDGE_TOOL.name,
                                      max_tokens=self.max_tokens)
        call = next((c for c in resp.tool_calls if c.name == JUDGE_TOOL.name), None)
        if call is None:
            raise ValueError("judge model did not call submit_judgments")
        out: list[Judgment] = []
        for item in call.arguments.get("judgments", []):
            try:
                idx, p = int(item["index"]), float(item["p_relevant"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= idx < len(hits):
                why = item.get("rationale")
                out.append(Judgment(key=hits[idx].key, p_relevant=_clamp(p),
                                    rationale=str(why) if why is not None else None))
        return JudgeResult(judgments=out, usage=resp.usage)

    async def decide(self, view: ControllerView) -> Decision:
        history = "\n".join(f"turn {t.turn}: {t.n_calls} calls, {t.n_errors} errors, {t.n_new} new, "
                            f"{t.n_new_relevant} new relevant" for t in view.history)
        text = (f"Question: {view.question.as_text()}\n\nTurns so far:\n{history}\n\n"
                f"Total judged relevant: {view.total_relevant}\n"
                f"Budget remaining: {json.dumps(view.budget_remaining)}\n\n"
                f"Latest digest:\n{view.digest}")
        resp = await self.client.chat(DECIDE_SYSTEM, [ChatMessage(role="user", content=text)],
                                      tools=[DECISION_TOOL], tool_choice=DECISION_TOOL.name,
                                      max_tokens=self.max_tokens)
        call = next((c for c in resp.tool_calls if c.name == DECISION_TOOL.name), None)
        if call is None:
            raise ValueError("decider model did not call submit_decision")
        return Decision(action=Action(str(call.arguments["action"]).lower()),
                        confidence=_clamp(float(call.arguments.get("confidence", 1.0))),
                        note=call.arguments.get("note"), usage=resp.usage)
```

- [ ] **Step 4: Implement the cross-encoder judge**

`src/agentic_search/models/cross_encoder.py`:
```python
"""Local cross-encoder/reranker as a relevance judge. Needs the `local` extra unless a scorer
is injected. Most 1-label rerankers (e.g. bge-reranker) already output [0, 1] via
sentence-transformers; set apply_sigmoid=True for models that return raw logits."""

from __future__ import annotations

import asyncio
import math
from typing import Any, Callable

from agentic_search.core.types import Hit, Query
from agentic_search.models.base import ControllerView, Decision, JudgeResult, Judgment

Scorer = Callable[[list[tuple[str, str]]], Any]


class CrossEncoderJudge:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", *, scorer: Scorer | None = None,
                 snippet_chars: int = 2000, apply_sigmoid: bool = False, id: str | None = None):
        self.model_name = model_name
        self._scorer = scorer
        self.snippet_chars = snippet_chars
        self.apply_sigmoid = apply_sigmoid
        self.id = id or f"cross-encoder:{model_name}"

    def _default_scorer(self) -> Scorer:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(self.model_name)
        return lambda pairs: model.predict(pairs)

    async def judge(self, question: Query, hits: list[Hit]) -> JudgeResult:
        if not hits:
            return JudgeResult(judgments=[])
        if self._scorer is None:
            self._scorer = await asyncio.to_thread(self._default_scorer)
        pairs = [(question.as_text(), h.snippet(self.snippet_chars)) for h in hits]
        scores = await asyncio.to_thread(self._scorer, pairs)
        out = []
        for h, raw in zip(hits, scores):
            s = float(raw)
            p = 1.0 / (1.0 + math.exp(-s)) if self.apply_sigmoid else min(1.0, max(0.0, s))
            out.append(Judgment(key=h.key, p_relevant=p))
        return JudgeResult(judgments=out)

    async def decide(self, view: ControllerView) -> Decision:
        raise NotImplementedError
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/models -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(models): LLM judge/decider and cross-encoder judge

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 18: Config loader

**Files:**
- Create: `src/agentic_search/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `ConfigError(ValueError)` and `BuildContext(base_dir, embedders)`.
  - `register(kind: "backend"|"embedder"|"client"|"decider", type_: str, factory: Callable[[dict, BuildContext], Any])`. Plans 2 and 3 call this.
  - `resolve_env(cfg)`: a `foo_env: VAR` key becomes `foo: os.environ[VAR]`.
  - `build_harness(cfg: dict, *, base_dir=None) -> Harness` and `load_harness(path) -> Harness`.
  - Built-in types:
    - backends: `files`
    - embedders: `hash`, `sentence_transformers`
    - clients (usable as `driver:`): `anthropic`, `openai_compat`
    - deciders: `cross_encoder`, `llm_judge` (takes a nested `client:`)
  - A backend's `allowed_models: [...]` feeds `source_policy`.

- [ ] **Step 1: Write the failing tests**

`tests/test_config.py`:
```python
import pytest

from agentic_search.config import ConfigError, build_harness, load_harness, register, resolve_env
from agentic_search.models.cross_encoder import CrossEncoderJudge
from agentic_search.models.llm_judge import LLMJudge


def test_resolve_env(monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret")
    assert resolve_env({"a": {"api_key_env": "MY_KEY"}, "b": [1]}) == {"a": {"api_key": "secret"}, "b": [1]}
    monkeypatch.delenv("MY_KEY")
    with pytest.raises(ConfigError, match="MY_KEY"):
        resolve_env({"api_key_env": "MY_KEY"})


def cfg(tmp_path):
    (tmp_path / "notes").mkdir(exist_ok=True)
    (tmp_path / "notes" / "a.md").write_text("aspirin headache")
    return {
        "embedders": [{"type": "hash", "id": "h", "dim": 32}],
        "backends": [{"name": "notes", "type": "files", "root": "notes", "embedder": "h",
                      "allowed_models": ["openai:local"]}],
        "driver": {"type": "openai_compat", "model": "local", "base_url": "http://localhost:8000/v1",
                   "max_calls_per_turn": 4},
        "analyzer": {"type": "cross_encoder", "model": "BAAI/bge-reranker-v2-m3"},
        "controller": {"type": "llm_judge", "client": {"type": "openai_compat", "model": "local",
                                                       "base_url": "http://localhost:8000/v1"}},
        "mode": "retrieval",
        "budget": {"max_turns": 2},
        "settings": {"judge_weight": 0.8},
    }


async def test_build_harness(tmp_path):
    h = build_harness(cfg(tmp_path), base_dir=tmp_path)
    assert h.driver.id == "openai:local" and h.driver.max_calls == 4
    assert isinstance(h.analyzer_decider, CrossEncoderJudge)
    assert isinstance(h.controller_decider, LLMJudge)
    assert h.mode == "retrieval" and h.budget.max_turns == 2 and h.settings.judge_weight == 0.8
    assert h.policy.allows("notes", "openai:local") and not h.policy.allows("notes", "other")
    manifests = await h.setup()
    assert manifests["notes"].resolve_collection(None).field("embedding").embedder_id == "h"


def test_errors(tmp_path):
    bad = cfg(tmp_path)
    bad["backends"][0]["type"] = "nope"
    with pytest.raises(ConfigError, match="unknown backend type 'nope'"):
        build_harness(bad, base_dir=tmp_path)
    no_driver = cfg(tmp_path)
    del no_driver["driver"]
    with pytest.raises(ConfigError, match="driver"):
        build_harness(no_driver, base_dir=tmp_path)
    missing_emb = cfg(tmp_path)
    missing_emb["backends"][0]["embedder"] = "zzz"
    with pytest.raises(ConfigError, match="zzz"):
        build_harness(missing_emb, base_dir=tmp_path)


async def test_load_yaml_with_annotations_file(tmp_path):
    import yaml
    c = cfg(tmp_path)
    c["annotations"] = "ann.yaml"
    (tmp_path / "ann.yaml").write_text(yaml.safe_dump({"notes": {"description": "Clinic notes"}}))
    (tmp_path / "h.yaml").write_text(yaml.safe_dump(c))
    h = load_harness(tmp_path / "h.yaml")
    await h.setup()
    assert h.manifests["notes"].description == "Clinic notes"


def test_register_custom_type(tmp_path):
    made = []
    register("embedder", "custom-test", lambda c, ctx: made.append(c) or __import__(
        "agentic_search.embedders.local", fromlist=["HashEmbedder"]).HashEmbedder(id="custom"))
    c = cfg(tmp_path)
    c["embedders"].append({"type": "custom-test", "x": 1})
    build_harness(c, base_dir=tmp_path)
    assert made == [{"type": "custom-test", "x": 1}]
    with pytest.raises(ValueError):
        register("gizmo", "x", lambda c, ctx: None)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_config.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`src/agentic_search/config.py`:
```python
"""Build a Harness from a YAML/dict config. Plans 2 and 3 register more types via register()."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from agentic_search.core.harness import Harness, HarnessSettings
from agentic_search.core.types import Budget
from agentic_search.embedders.base import Embedder


class ConfigError(ValueError):
    pass


@dataclass
class BuildContext:
    base_dir: Path
    embedders: dict[str, Embedder] = field(default_factory=dict)


Factory = Callable[[dict[str, Any], BuildContext], Any]
_REGISTRIES: dict[str, dict[str, Factory]] = {"backend": {}, "embedder": {}, "client": {}, "decider": {}}


def register(kind: str, type_: str, factory: Factory) -> None:
    if kind not in _REGISTRIES:
        raise ValueError(f"unknown registry kind {kind!r}; one of {sorted(_REGISTRIES)}")
    _REGISTRIES[kind][type_] = factory


def resolve_env(cfg: Any) -> Any:
    if isinstance(cfg, dict):
        out: dict[str, Any] = {}
        for k, v in cfg.items():
            if isinstance(k, str) and k.endswith("_env") and isinstance(v, str):
                if v not in os.environ:
                    raise ConfigError(f"environment variable {v!r} (for {k!r}) is not set")
                out[k[: -len("_env")]] = os.environ[v]
            else:
                out[k] = resolve_env(v)
        return out
    if isinstance(cfg, list):
        return [resolve_env(v) for v in cfg]
    return cfg


def build(kind: str, cfg: dict[str, Any], ctx: BuildContext) -> Any:
    cfg = resolve_env(cfg)
    type_ = cfg.get("type")
    factory = _REGISTRIES[kind].get(type_)  # type: ignore[arg-type]
    if factory is None:
        raise ConfigError(f"unknown {kind} type {type_!r}; known: {sorted(_REGISTRIES[kind])}")
    return factory(cfg, ctx)


def _price(cfg: dict[str, Any]) -> tuple[float, float] | None:
    p = cfg.get("price_per_mtok")
    return (float(p[0]), float(p[1])) if p else None


# ---- built-in factories (imports are lazy so optional extras stay optional) ----

def _files(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.backends.files import FilesBackend
    embedder = None
    if "embedder" in cfg:
        embedder = ctx.embedders.get(cfg["embedder"])
        if embedder is None:
            raise ConfigError(f"backend {cfg['name']!r} references unknown embedder {cfg['embedder']!r}")
    root = Path(cfg["root"])
    return FilesBackend(cfg["name"], root if root.is_absolute() else ctx.base_dir / root,
                        embedder=embedder, glob=cfg.get("glob", "**/*"),
                        collection=cfg.get("collection", "files"),
                        description=cfg.get("description"))


def _hash(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.embedders.local import HashEmbedder
    return HashEmbedder(dim=int(cfg.get("dim", 256)), id=cfg.get("id", "hash"))


def _sentence_transformers(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.embedders.local import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder(cfg["model"], id=cfg.get("id"), image=cfg.get("image", False),
                                       query_prefix=cfg.get("query_prefix", ""),
                                       document_prefix=cfg.get("document_prefix", ""))


def _anthropic(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.models.anthropic import AnthropicClient
    return AnthropicClient(cfg["model"], api_key=cfg.get("api_key"), id=cfg.get("id"),
                           supports_images=cfg.get("supports_images", True), price_per_mtok=_price(cfg))


def _openai_compat(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.models.openai_compat import OpenAICompatClient
    return OpenAICompatClient(cfg["model"], base_url=cfg.get("base_url"), api_key=cfg.get("api_key"),
                              id=cfg.get("id"), supports_images=cfg.get("supports_images", False),
                              price_per_mtok=_price(cfg),
                              max_tokens_param=cfg.get("max_tokens_param", "max_tokens"))


def _cross_encoder(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.models.cross_encoder import CrossEncoderJudge
    return CrossEncoderJudge(cfg.get("model", "BAAI/bge-reranker-v2-m3"),
                             apply_sigmoid=cfg.get("apply_sigmoid", False), id=cfg.get("id"))


def _llm_judge(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.models.llm_judge import LLMJudge
    if "client" not in cfg:
        raise ConfigError("llm_judge needs a nested `client:` config")
    return LLMJudge(build("client", cfg["client"], ctx), id=cfg.get("id"))


for _kind, _type, _factory in [
    ("backend", "files", _files),
    ("embedder", "hash", _hash),
    ("embedder", "sentence_transformers", _sentence_transformers),
    ("client", "anthropic", _anthropic),
    ("client", "openai_compat", _openai_compat),
    ("decider", "cross_encoder", _cross_encoder),
    ("decider", "llm_judge", _llm_judge),
]:
    register(_kind, _type, _factory)


def build_harness(cfg: dict[str, Any], *, base_dir: Path | None = None) -> Harness:
    from agentic_search.models.driver import ToolCallingDriver

    ctx = BuildContext(base_dir=Path(base_dir) if base_dir else Path.cwd())
    for ecfg in cfg.get("embedders", []):
        embedder = build("embedder", ecfg, ctx)
        ctx.embedders[embedder.id] = embedder
    backends, policy = [], {}
    for bcfg in cfg.get("backends", []):
        backends.append(build("backend", bcfg, ctx))
        if "allowed_models" in bcfg:
            policy[bcfg["name"]] = set(bcfg["allowed_models"])
    if not cfg.get("driver"):
        raise ConfigError("config needs a `driver:` section")
    driver_cfg = dict(cfg["driver"])
    max_calls = int(driver_cfg.pop("max_calls_per_turn", 8))
    driver = ToolCallingDriver(build("client", driver_cfg, ctx), max_calls_per_turn=max_calls)
    analyzer = build("decider", cfg["analyzer"], ctx) if cfg.get("analyzer") else None
    controller = build("decider", cfg["controller"], ctx) if cfg.get("controller") else None
    annotations = cfg.get("annotations")
    if isinstance(annotations, str):
        annotations = yaml.safe_load((ctx.base_dir / annotations).read_text()) or {}
    return Harness(backends, driver, embedders=list(ctx.embedders.values()), analyzer=analyzer,
                   controller=controller, mode=cfg.get("mode", "harness"),
                   budget=Budget(**cfg.get("budget", {})), source_policy=policy or None,
                   annotations=annotations, settings=HarnessSettings(**cfg.get("settings", {})))


def load_harness(path: str | Path) -> Harness:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    return build_harness(cfg, base_dir=path.parent)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_config.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: YAML config loader with pluggable type registries

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---
### Task 19: BEIR eval, comparison script, README

**Files:**
- Create: `src/agentic_search/eval/datasets.py`, `src/agentic_search/eval/metrics.py`, `src/agentic_search/eval/runner.py`
- Create: `scripts/eval_beir.py`, `README.md`
- Test: `tests/eval/test_eval.py` (plus an empty `tests/eval/__init__.py`)

**Interfaces:**
- Consumes: `Harness` (Task 13), `EchoDriver` (Task 6), `FilesBackend` (Task 5), and the clients and judges (Tasks 14–17).
- Produces:
  - `BeirDataset(name, corpus, queries, qrels)` with `.documents() -> list[Document]`. `load_beir(path, split="test")` keeps only the queries that have qrels. `download_beir(name, dest_dir) -> Path`.
  - Metrics (linear gain, matching BEIR/pytrec_eval): `ndcg_at_k(ranked, qrels, k)`, `recall_at_k(ranked, qrels, k)` and `mrr(ranked, qrels)`.
  - `QueryRun` and `EvalReport` (`.mean(attr)`, `.table()`).
  - `async run_eval(harness, dataset, *, name, query_ids=None, top_k=100, k=10, recall_k=100, concurrency=4, mode=None) -> EvalReport`.

- [ ] **Step 1: Write the failing tests**

`tests/eval/test_eval.py`:
```python
import json

import pytest

from agentic_search import Harness
from agentic_search.backends.files import FilesBackend
from agentic_search.eval.datasets import load_beir
from agentic_search.eval.metrics import mrr, ndcg_at_k, recall_at_k
from agentic_search.eval.runner import run_eval
from agentic_search.testing import EchoDriver


def test_metrics():
    assert ndcg_at_k(["a", "b"], {"a": 1}, 10) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "a"], {"a": 1}, 10) == pytest.approx(0.6309, abs=1e-4)
    assert ndcg_at_k(["x"], {}, 10) == 0.0
    assert recall_at_k(["a", "x"], {"a": 1, "b": 2, "c": 0}, 1) == 0.5
    assert mrr(["x", "a"], {"a": 1}) == 0.5 and mrr(["x"], {"a": 1}) == 0.0


@pytest.fixture
def beir_dir(tmp_path):
    corpus = [{"_id": "d1", "title": "", "text": "aspirin headache"},
              {"_id": "d2", "title": "Castles", "text": "castle history"},
              {"_id": "d3", "title": "", "text": "ibuprofen headache pain"}]
    queries = [{"_id": "q1", "text": "headache"}, {"_id": "q2", "text": "castle"},
               {"_id": "q3", "text": "no qrels"}]
    (tmp_path / "corpus.jsonl").write_text("\n".join(json.dumps(r) for r in corpus))
    (tmp_path / "queries.jsonl").write_text("\n".join(json.dumps(r) for r in queries))
    (tmp_path / "qrels").mkdir()
    (tmp_path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td3\t2\nq2\td2\t1\n")
    return tmp_path


def test_load_beir(beir_dir):
    ds = load_beir(beir_dir)
    assert set(ds.queries) == {"q1", "q2"} and ds.qrels["q1"] == {"d1": 1, "d3": 2}
    docs = {d.doc_id: d for d in ds.documents()}
    assert docs["d2"].content[0].text == "Castles\ncastle history"
    assert docs["d2"].metadata == {"title": "Castles"} and docs["d1"].metadata == {}


async def test_run_eval_bm25_baseline(beir_dir):
    ds = load_beir(beir_dir)
    backend = FilesBackend.from_documents("corpus", ds.documents())
    h = Harness([backend], EchoDriver("corpus"), mode="retrieval")
    report = await run_eval(h, ds, name="bm25")
    runs = {r.query_id: r for r in report.runs}
    # q1 ranks d1 (short doc) above d3: DCG = 1 + 2/log2(3); IDCG = 2 + 1/log2(3)
    assert runs["q1"].ndcg == pytest.approx(0.8597, abs=1e-3)
    assert runs["q2"].ndcg == pytest.approx(1.0) and runs["q2"].recall == 1.0
    assert report.mean("ndcg") == pytest.approx((0.8597 + 1.0) / 2, abs=1e-3)
    assert "bm25" in report.table() and "nDCG@10" in report.table()
    assert all(r.error is None for r in report.runs)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/eval -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement metrics**

`src/agentic_search/eval/metrics.py`:
```python
"""IR metrics with linear gain, matching BEIR / pytrec_eval."""

from __future__ import annotations

import math


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    ideal = _dcg(sorted((float(r) for r in qrels.values() if r > 0), reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return _dcg([float(max(qrels.get(d, 0), 0)) for d in ranked[:k]]) / ideal


def recall_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    relevant = {d for d, r in qrels.items() if r > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def mrr(ranked: list[str], qrels: dict[str, int]) -> float:
    for i, d in enumerate(ranked):
        if qrels.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0
```

- [ ] **Step 4: Implement datasets**

`src/agentic_search/eval/datasets.py`:
```python
"""BEIR-format datasets: corpus.jsonl, queries.jsonl, qrels/<split>.tsv."""

from __future__ import annotations

import csv
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from agentic_search.core.types import Document, TextPart

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


@dataclass
class BeirDataset:
    name: str
    corpus: dict[str, dict[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]

    def documents(self) -> list[Document]:
        docs = []
        for doc_id, row in self.corpus.items():
            title, text = row.get("title", ""), row.get("text", "")
            docs.append(Document(doc_id=doc_id, content=[TextPart(text=f"{title}\n{text}".strip())],
                                 metadata={"title": title} if title else {}))
        return docs


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_beir(path: str | Path, split: str = "test") -> BeirDataset:
    path = Path(path)
    corpus = {str(r["_id"]): {"title": r.get("title") or "", "text": r.get("text") or ""}
              for r in _jsonl(path / "corpus.jsonl")}
    qrels: dict[str, dict[str, int]] = {}
    with (path / "qrels" / f"{split}.tsv").open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # header
        for row in reader:
            if len(row) >= 3:
                qrels.setdefault(row[0], {})[row[1]] = int(row[2])
    queries = {str(r["_id"]): r["text"] for r in _jsonl(path / "queries.jsonl")
               if str(r["_id"]) in qrels}
    return BeirDataset(name=path.name, corpus=corpus, queries=queries, qrels=qrels)


def download_beir(name: str, dest_dir: str | Path) -> Path:
    dest = Path(dest_dir)
    target = dest / name
    if (target / "corpus.jsonl").exists():
        return target
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / f"{name}.zip"
    urllib.request.urlretrieve(BEIR_URL.format(name=name), archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest)
    archive.unlink()
    return target
```

- [ ] **Step 5: Implement the runner**

`src/agentic_search/eval/runner.py`:
```python
"""Run a Harness over a labeled dataset and report quality, cost and latency."""

from __future__ import annotations

import asyncio
import time

from pydantic import BaseModel

from agentic_search.core.harness import Harness, Mode
from agentic_search.eval.datasets import BeirDataset
from agentic_search.eval.metrics import mrr, ndcg_at_k, recall_at_k


class QueryRun(BaseModel):
    query_id: str
    ndcg: float
    recall: float
    mrr: float
    cost_usd: float
    seconds: float
    tool_calls: int
    stop_reason: str
    error: str | None = None


class EvalReport(BaseModel):
    name: str
    k: int
    recall_k: int
    runs: list[QueryRun]

    def mean(self, attr: str) -> float:
        values = [float(getattr(r, attr)) for r in self.runs]
        return sum(values) / len(values) if values else 0.0

    def table(self) -> str:
        errors = sum(r.error is not None for r in self.runs)
        return (f"{self.name:<28} nDCG@{self.k}={self.mean('ndcg'):.4f} "
                f"R@{self.recall_k}={self.mean('recall'):.4f} MRR={self.mean('mrr'):.4f} "
                f"cost=${self.mean('cost_usd'):.4f}/q latency={self.mean('seconds'):.2f}s/q "
                f"calls={self.mean('tool_calls'):.1f}/q n={len(self.runs)} errors={errors}")


async def run_eval(harness: Harness, dataset: BeirDataset, *, name: str,
                   query_ids: list[str] | None = None, top_k: int = 100, k: int = 10,
                   recall_k: int = 100, concurrency: int = 4, mode: Mode | None = None) -> EvalReport:
    qids = query_ids or sorted(dataset.queries)
    sem = asyncio.Semaphore(concurrency)

    async def one(qid: str) -> QueryRun:
        async with sem:
            t0 = time.perf_counter()
            try:
                res = await harness.search(dataset.queries[qid], top_k=top_k, mode=mode)
            except Exception as exc:
                return QueryRun(query_id=qid, ndcg=0.0, recall=0.0, mrr=0.0, cost_usd=0.0,
                                seconds=time.perf_counter() - t0, tool_calls=0, stop_reason="error",
                                error=f"{type(exc).__name__}: {exc}")
            ranked = [h.hit.doc_id for h in res.hits]
            qrels = dataset.qrels.get(qid, {})
            return QueryRun(query_id=qid, ndcg=ndcg_at_k(ranked, qrels, k),
                            recall=recall_at_k(ranked, qrels, recall_k), mrr=mrr(ranked, qrels),
                            cost_usd=res.usage.cost_usd, seconds=time.perf_counter() - t0,
                            tool_calls=res.usage.tool_calls, stop_reason=res.stop_reason.value)

    runs = await asyncio.gather(*(one(q) for q in qids))
    return EvalReport(name=name, k=k, recall_k=recall_k, runs=list(runs))
```

- [ ] **Step 6: Run to verify pass**

Run: `uv run pytest tests/eval -q`
Expected: all pass

- [ ] **Step 7: Write the comparison script**

`scripts/eval_beir.py`:
```python
"""Compare harness modes on a BEIR dataset (spec success criterion 3).

Examples:
  uv run python scripts/eval_beir.py --modes bm25 --limit 50
  uv run --extra local python scripts/eval_beir.py --embedder st:BAAI/bge-small-en-v1.5 \
      --driver anthropic:claude-sonnet-5-5 --judge llm --modes bm25,retrieval,harness,model --limit 50
  uv run python scripts/eval_beir.py --driver openai:Qwen/Qwen3-8B --modes harness  # OPENAI_BASE_URL=...
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from agentic_search import Budget, Harness
from agentic_search.backends.files import FilesBackend
from agentic_search.eval.datasets import download_beir, load_beir
from agentic_search.eval.runner import run_eval
from agentic_search.testing import EchoDriver


def make_client(spec: str):
    provider, _, model = spec.partition(":")
    if provider == "anthropic":
        from agentic_search.models.anthropic import AnthropicClient
        return AnthropicClient(model)
    if provider == "openai":
        from agentic_search.models.openai_compat import OpenAICompatClient
        return OpenAICompatClient(model, base_url=os.environ.get("OPENAI_BASE_URL"))
    raise SystemExit(f"unknown driver provider {provider!r} (use anthropic:<model> or openai:<model>)")


def make_embedder(spec: str):
    if spec == "none":
        return None
    if spec == "hash":
        from agentic_search.embedders.local import HashEmbedder
        return HashEmbedder()
    if spec.startswith("st:"):
        from agentic_search.embedders.local import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(spec[3:])
    raise SystemExit(f"unknown embedder {spec!r} (none | hash | st:<model>)")


def make_judge(spec: str, client):
    if spec == "none":
        return None
    if spec == "llm":
        from agentic_search.models.llm_judge import LLMJudge
        return LLMJudge(client)
    if spec.startswith("cross_encoder"):
        from agentic_search.models.cross_encoder import CrossEncoderJudge
        _, _, model = spec.partition(":")
        return CrossEncoderJudge(model or "BAAI/bge-reranker-v2-m3")
    raise SystemExit(f"unknown judge {spec!r} (none | llm | cross_encoder[:model])")


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="nfcorpus")
    p.add_argument("--data-dir", default="data/beir")
    p.add_argument("--modes", default="bm25,retrieval,harness")
    p.add_argument("--driver", default="anthropic:claude-sonnet-5-5")
    p.add_argument("--judge", default="llm")
    p.add_argument("--embedder", default="none")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--max-turns", type=int, default=4)
    p.add_argument("--max-tool-calls", type=int, default=24)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--out", default="eval-results")
    args = p.parse_args()

    ds = load_beir(download_beir(args.dataset, args.data_dir))
    qids = sorted(ds.queries)[: args.limit]
    embedder = make_embedder(args.embedder)
    backend = FilesBackend.from_documents(
        "corpus", ds.documents(), embedder=embedder,
        description=f"BEIR {ds.name} corpus: one document per title + abstract.")
    budget = Budget(max_turns=args.max_turns, max_tool_calls=args.max_tool_calls, max_seconds=300)
    embedders = [embedder] if embedder else []
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = None
    for mode in args.modes.split(","):
        if mode == "bm25":
            harness = Harness([backend], EchoDriver("corpus"), mode="retrieval")
        else:
            from agentic_search.models.driver import ToolCallingDriver
            client = client or make_client(args.driver)
            harness = Harness([backend], ToolCallingDriver(client), embedders=embedders,
                              analyzer=make_judge(args.judge, client), mode=mode, budget=budget)
        report = await run_eval(harness, ds, name=f"{ds.name}/{mode}", query_ids=qids,
                                concurrency=args.concurrency)
        print(report.table(), flush=True)
        (out / f"{ds.name}-{mode}.json").write_text(report.model_dump_json(indent=1))


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 8: Run the offline baseline end to end**

Run: `uv run python scripts/eval_beir.py --modes bm25 --limit 50`
Expected: NFCorpus downloads to `data/beir/nfcorpus` and one line prints, e.g. `nfcorpus/bm25 nDCG@10=0.2… … errors=0`. Record the exact number in the commit message. BM25 on NFCorpus is usually around 0.3 nDCG@10. A value near 0 means ids or qrels are wired wrong, so stop and debug.

- [ ] **Step 9: Write the README**

`README.md`:
````markdown
# agentic-search

An agentic search harness: **plan → execute → analyze → decide** over any datastore, with swappable
models. Modeled on Doug Turnbull's *Three Kinds of Agentic Search* and SID.ai's SID-1.
Design: `docs/superpowers/specs/2026-09-29-agentic-search-harness-design.md`.

## Quickstart

```bash
uv sync --extra anthropic
export ANTHROPIC_API_KEY=...
```

```python
import asyncio
from agentic_search import Harness
from agentic_search.backends.files import FilesBackend
from agentic_search.embedders.local import HashEmbedder
from agentic_search.models.anthropic import AnthropicClient
from agentic_search.models.driver import ToolCallingDriver
from agentic_search.models.llm_judge import LLMJudge

async def main():
    client = AnthropicClient("claude-sonnet-5-5")
    emb = HashEmbedder()
    notes = FilesBackend("notes", "./my-notes", embedder=emb)
    async with Harness([notes], ToolCallingDriver(client), embedders=[emb],
                       analyzer=LLMJudge(client)) as h:
        result = await h.search("what did we decide about the Q3 launch?")
        for r in result.hits[:5]:
            print(f"{r.score:.2f} {r.hit.key} {r.hit.snippet(100)}")
        print(result.stop_reason, result.usage)

asyncio.run(main())
```

Or from YAML: `from agentic_search.config import load_harness`. The spec §7 shows the format.

## Modes

| mode | what happens |
|---|---|
| `retrieval` | one planned pass, rank by fused backend scores (optionally judged) |
| `harness` | full loop; the judge's feedback steers the next turn's plan |
| `model` | a trained search model (e.g. SID-1 via an OpenAI-compatible endpoint) runs its own tool loop; the harness enforces budgets and records the trace |

## Roles

- **Driver**: plans and calls tools (`ToolCallingDriver` over `AnthropicClient` or `OpenAICompatClient`).
- **Analyzer decider**: judges relevance (`LLMJudge`, `CrossEncoderJudge`, TypeSafe System One in Plan 3).
- **Controller decider**: continue/refine/broaden/stop (`LLMJudge`, or the built-in heuristic).
- **Backends**: `FilesBackend` now. SQL, OpenSearch, graph and vector stores come in Plans 2–3.
- **Hooks / SourcePolicy**: every model-bound payload passes through `Hooks.before_model_call`;
  per-source `allowed_models` withholds raw content from other models.

## Evaluate

```bash
uv run python scripts/eval_beir.py --modes bm25 --limit 50                      # offline baseline
uv run python scripts/eval_beir.py --modes bm25,retrieval,harness --limit 50    # needs ANTHROPIC_API_KEY
```

## Develop

```bash
uv sync && uv run pytest
```
````

- [ ] **Step 10: Full suite + lint**

Run: `uv run ruff check --fix src tests scripts && uv run pytest -q && uv run ruff check src tests scripts`
Expected: `--fix` sorts imports (I001); then all tests pass and ruff reports `All checks passed!`.

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "feat(eval): BEIR loader, metrics, runner, mode-comparison script, README

NFCorpus BM25 baseline (50 queries): nDCG@10=<value from step 8>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Spec coverage (Plan 1)

| Spec § | Where |
|---|---|
| 3.2 core types | Task 2 |
| 3.3 backend protocol, discover, annotations | Tasks 5, 12 (SQL/search/graph/vector adapters → Plans 2–3) |
| 3.4 embedders + space binding | Tasks 4, 9 (remote adapters + auth → Plan 3) |
| 3.5 driver/decider roles | Tasks 6, 14–17 (TypeSafe → Plan 3) |
| 4.0–4.6 loop and modes | Tasks 8–13 |
| 5 trace | Tasks 7, 9–13 |
| 6 errors, fail-open, budgets, hooks, source policy | Tasks 7, 9–11, 13 |
| 6 native-query guard, BigQuery byte cap | Plan 2 |
| 7 configuration | Task 18 |
| 8 testing, minimal eval | every task; Task 19 |
| 9 success criteria 2, 4 | Tasks 13, 9 (`test_vector_refuses_unregistered_embedder`) |
| 9 success criterion 3 | Task 19 script (live run needs an API key) |
| 9 criteria 1, 5 | Plans 2–3 |
