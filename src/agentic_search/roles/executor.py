"""Validates tool calls against manifests, embeds queries, runs ops in parallel, pools hits."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from agentic_search.backends.base import Backend
from agentic_search.backends.filters import filter_fields
from agentic_search.core.emitter import EventEmitter, NullEmitter
from agentic_search.core.hooks import Hooks, SourcePolicy
from agentic_search.core.secrets import scrub
from agentic_search.core.state import CandidatePool, Trace
from agentic_search.core.types import (
    Aggregate,
    FieldType,
    Hit,
    Hybrid,
    Lexical,
    Manifest,
    Query,
    QueryOp,
    Regex,
    TextPart,
    ToolError,
    Vector,
    modality_of,
    required_capabilities,
)
from agentic_search.embedders.base import EmbedderRegistry
from agentic_search.events import ToolCallFinished, ToolCallStarted, ToolErrorInfo
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
    return scrub(str(exc))[:500]


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
                  trace: Trace, model_id: str | None = None,
                  emitter: EventEmitter | None = None) -> ExecResult:
        em = emitter or NullEmitter()
        outcomes = await asyncio.gather(*(self._run_emitting(c, question, turn, trace, em)
                                          for c in calls))
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

    async def _run_emitting(self, c: ToolCall, question: Query, turn: int, trace: Trace,
                            emitter: EventEmitter) -> _Outcome:
        source = c.arguments.get("source")
        emitter.emit(ToolCallStarted, turn=turn, call_id=c.id, name=c.name, arguments=c.arguments,
                     source=source if isinstance(source, str) else None)
        t0 = time.perf_counter()
        out = await self._run_one(c, question, turn, trace)
        err = out.error
        emitter.emit(ToolCallFinished, turn=turn, call_id=c.id, n_hits=len(out.hits),
                     duration_ms=(time.perf_counter() - t0) * 1000,
                     error=None if err is None else ToolErrorInfo(
                         kind=err.kind, message=scrub(err.message), source=err.source))
        return out

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
                op = await asyncio.wait_for(self._embed(op), self.call_timeout)
            except TimeoutError:
                return _Outcome(error=ToolError(call_id=c.id, source=op.source, kind="timeout",
                                                message=f"embedding timed out after {self.call_timeout}s"))
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
