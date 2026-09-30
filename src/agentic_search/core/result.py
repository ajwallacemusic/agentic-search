"""The final result of one search."""

from __future__ import annotations

from pydantic import BaseModel

from agentic_search.core.state import Trace, Usage
from agentic_search.core.types import Hit, Query, StopReason


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
