"""Typed, JSON-serialisable events for one streamed search (schema version 1).

`SearchEvent` is a discriminated union on `type`. Consumers must ignore event types they do not
know: adding an optional field or a new event type keeps `schema_version`; renaming or removing
a field, changing a type or changing ordering guarantees bumps it."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from agentic_search.core.result import SearchResult
from agentic_search.core.state import Usage
from agentic_search.core.types import Budget, Content, Query, StopReason

SCHEMA_VERSION = 1
Phase = Literal["plan", "query", "judge", "decide", "delegate"]


class _Event(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = SCHEMA_VERSION
    search_id: str
    seq: int
    turn: int
    at_ms: float


class PhaseSummary(BaseModel):
    """Per-phase counts; only the fields relevant to the phase are set."""

    model_config = ConfigDict(frozen=True)

    n_calls: int | None = None
    n_hits: int | None = None
    n_new: int | None = None
    n_errors: int | None = None
    n_judged: int | None = None
    n_relevant: int | None = None
    action: str | None = None
    confidence: float | None = None
    n_ranked: int | None = None
    note: str | None = None
    error: str | None = None


class ToolErrorInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: str
    message: str
    source: str | None = None


class HitSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    source: str
    doc_id: str
    title: str | None = None
    snippet: str
    score: float
    p_relevant: float | None = None
    judged: bool = False
    first_turn: int
    content: list[Content] | None = None


class SearchStarted(_Event):
    type: Literal["search_started"] = "search_started"
    question: Query
    mode: str
    sources: list[str]
    budget: Budget
    setup_errors: dict[str, str] = Field(default_factory=dict)


class PhaseStarted(_Event):
    type: Literal["phase_started"] = "phase_started"
    phase: Phase


class PhaseFinished(_Event):
    type: Literal["phase_finished"] = "phase_finished"
    phase: Phase
    duration_ms: float
    summary: PhaseSummary = Field(default_factory=PhaseSummary)


class ToolCallStarted(_Event):
    type: Literal["tool_call_started"] = "tool_call_started"
    call_id: str
    source: str | None = None
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCallFinished(_Event):
    type: Literal["tool_call_finished"] = "tool_call_finished"
    call_id: str
    n_hits: int
    duration_ms: float
    error: ToolErrorInfo | None = None


class ResultsUpdated(_Event):
    type: Literal["results_updated"] = "results_updated"
    hits: list[HitSummary]
    pool_size: int
    n_relevant: int


class UsageUpdated(_Event):
    type: Literal["usage_updated"] = "usage_updated"
    usage: Usage


class SearchFinished(_Event):
    type: Literal["search_finished"] = "search_finished"
    result: SearchResult
    stop_reason: StopReason


class SearchFailed(_Event):
    type: Literal["search_failed"] = "search_failed"
    error_type: str
    message: str


SearchEvent = Annotated[
    Union[SearchStarted, PhaseStarted, PhaseFinished, ToolCallStarted, ToolCallFinished,
          ResultsUpdated, UsageUpdated, SearchFinished, SearchFailed],
    Field(discriminator="type"),
]
TERMINAL_EVENTS = (SearchFinished, SearchFailed)
