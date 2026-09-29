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
