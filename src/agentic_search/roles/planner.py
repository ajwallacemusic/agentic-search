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
