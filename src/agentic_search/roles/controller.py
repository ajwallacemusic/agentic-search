"""Decides whether to keep searching. Hard budgets always win over any decider."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from agentic_search.core.hooks import Hooks
from agentic_search.core.secrets import scrub
from agentic_search.core.state import SearchState
from agentic_search.core.types import StopReason
from agentic_search.events import PhaseFinished, PhaseStarted, PhaseSummary
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

    async def decide(self, state: SearchState, *, digest: str | None = None) -> Decision:
        """`digest` is the turn digest rendered for this controller's decider (default state.digest)."""
        state.emitter.emit(PhaseStarted, turn=state.turn, phase="decide")
        t0 = time.perf_counter()
        by = "heuristic"
        error: str | None = None
        if self.decider is None or not self._can_decide:
            decision = self.heuristic(state)
        else:
            decider = self.decider
            view = ControllerView(question=state.question, turn=state.turn, history=state.history,
                                  digest=state.digest if digest is None else digest,
                                  total_relevant=self.total_relevant(state),
                                  budget_remaining=self.budget_remaining(state))
            try:
                view = await self.hooks.before_model_call(decider.id, view)
                decision = await asyncio.wait_for(decider.decide(view), self.timeout)
                by = decider.id
            except NotImplementedError:
                self._can_decide = False
                decision = self.heuristic(state)
            except Exception as exc:
                error = scrub(f"{type(exc).__name__}: {exc}")[:300]
                state.trace.add("decision_error", state.turn, decider=decider.id, error=error)
                decision = self.heuristic(state)
        state.trace.add("decision", state.turn, action=decision.action.value, note=decision.note,
                        confidence=decision.confidence, by=by)
        state.emitter.emit(PhaseFinished, turn=state.turn, phase="decide",
                           duration_ms=(time.perf_counter() - t0) * 1000,
                           summary=PhaseSummary(action=decision.action.value,
                                                confidence=decision.confidence, error=error))
        return decision
