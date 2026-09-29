"""Per-search mutable state: candidate pool, trace, usage."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, Field, PrivateAttr

from agentic_search.core.secrets import scrub_data
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
                           duration_ms=duration_ms, data=scrub_data(data))
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
