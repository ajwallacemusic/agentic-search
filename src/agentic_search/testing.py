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
