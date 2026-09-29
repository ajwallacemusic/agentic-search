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


class JudgeRequest(BaseModel):
    """What a judge receives; passed through Hooks.before_model_call."""

    question: Query
    hits: list[Hit]


class DelegateRequest(BaseModel):
    """What a delegate-mode driver receives up front; passed through Hooks.before_model_call."""

    question: Query
    context: str = ""


class ToolRuntime(Protocol):
    """Executes tool calls for a delegate-mode driver; returns one text result per call.
    Drivers call report_usage after every model response so cost/token budgets apply mid-run."""

    async def call(self, calls: list[ToolCall]) -> list[str]: ...

    def report_usage(self, usage: ModelUsage) -> None: ...

    def budget_exhausted(self) -> bool:
        """True once the harness has refused tool calls for budget reasons; drivers stop calling
        the model and finish."""
        ...


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
