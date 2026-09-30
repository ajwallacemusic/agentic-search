"""The Harness: Plan → Execute → Analyze → Decide over any backends, with swappable models."""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any, Literal, Sequence

from pydantic import BaseModel

from agentic_search.backends.base import Backend
from agentic_search.core.annotations import apply_annotations
from agentic_search.core.hooks import Hooks, SourcePolicy
from agentic_search.core.secrets import scrub
from agentic_search.core.state import Candidate, SearchState, Trace, Usage
from agentic_search.core.types import Budget, Hit, Manifest, ModelUsage, Query, StopReason
from agentic_search.embedders.base import Embedder, EmbedderRegistry
from agentic_search.models.base import (
    Action,
    Decider,
    DelegateRequest,
    Driver,
    ToolCall,
    ToolSpec,
    TurnSummary,
)
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
    discover_timeout: float | None = 300.0


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

    def __init__(self, state: SearchState, executor: Executor, controller: Controller, model_id: str,
                 hooks: Hooks):
        self.state, self.executor, self.controller, self.model_id = state, executor, controller, model_id
        self.hooks = hooks
        self.exhausted: StopReason | None = None
        self.reported = ModelUsage()

    def report_usage(self, usage: ModelUsage) -> None:
        self.state.usage.add_model(usage)
        self.reported = self.reported.plus(usage)

    def budget_exhausted(self) -> bool:
        reason = self.exhausted or self.controller.budget_stop(self.state)
        if reason is not None and self.exhausted is None:
            self.exhausted = reason
        return reason is not None

    def unreported(self, total: ModelUsage) -> ModelUsage:
        """The part of a driver's final usage total not already reported mid-run."""
        r = self.reported
        return ModelUsage(input_tokens=max(0, total.input_tokens - r.input_tokens),
                          output_tokens=max(0, total.output_tokens - r.output_tokens),
                          cost_usd=max(0.0, total.cost_usd - r.cost_usd))

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
        texts = [outputs.get(c.id, message) for c in calls]
        try:
            return list(await self.hooks.before_model_call(self.model_id, texts))
        except Exception as exc:
            state.trace.add("hook_error", state.turn, model=self.model_id, n=len(calls),
                            error=f"{type(exc).__name__}: {str(exc)[:300]}")
            return [f"[withheld by hook] {type(exc).__name__}" for _ in calls]


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
        for backend in backends:  # backends that embed their own content get hooked embedders
            bind = getattr(backend, "bind_hooks", None)
            if callable(bind):
                bind(self.hooks, self.policy)
        self.manifests: dict[str, Manifest] = {}
        self.setup_errors: dict[str, str] = {}
        self._ready = False
        self._setup_lock = asyncio.Lock()

    async def setup(self) -> dict[str, Manifest]:
        async with self._setup_lock:
            if self._ready:
                return self.manifests
            names = list(self.backends)
            timeout = self.settings.discover_timeout
            results = await asyncio.gather(
                *(asyncio.wait_for(self.backends[n].discover("full"), timeout) for n in names),
                return_exceptions=True)
            for name, res in zip(names, results):
                if isinstance(res, BaseException):
                    self.setup_errors[name] = scrub(f"{type(res).__name__}: {res}")
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
            reason = controller.budget_stop(state)
            if reason is not None:
                state.trace.add("budget", state.turn, reason=reason.value)
                return reason
            ctrl_digest = None
            if self.controller_decider is not None:
                ctrl_digest = analyzer.render_digest(state, calls, result, self.controller_decider.id)
            decision = await controller.decide(state, digest=ctrl_digest)
            state.usage.add_model(decision.usage)
            if decision.action is Action.STOP:
                return StopReason.CONTROLLER_STOP
            state.last_decision = decision
            state.turn += 1

    async def _run_delegate(self, state: SearchState, executor: Executor, analyzer: Analyzer,
                            controller: Controller,
                            tools: list[ToolSpec]) -> tuple[StopReason, list[str] | None]:
        runtime = _DelegateRuntime(state, executor, controller, self.driver.id, self.hooks)
        request = DelegateRequest(question=state.question, context=render_manifests(state.manifests))
        request = await self.hooks.before_model_call(self.driver.id, request)
        t0 = time.perf_counter()
        result = await self.driver.run_delegate(request.question, tools, runtime, state.budget,
                                                context=request.context)
        state.usage.add_model(runtime.unreported(result.usage))
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

    def _closeables(self) -> list[Any]:
        """Backends, plus embedders, deciders, the driver and the objects they wrap (`inner`) or
        hold (`client`), deduplicated by identity, that expose an async `close()`. Backends own their
        clients and close them themselves, so they are not traversed."""
        seen: dict[int, Any] = {id(b): b for b in self.backends.values()}
        roots: list[Any] = [*(self.embedders.get(i) for i in self.embedders.ids()),
                            self.analyzer_decider, self.controller_decider, self.driver]
        while roots:
            obj = roots.pop()
            if obj is None or id(obj) in seen:
                continue
            seen[id(obj)] = obj
            roots.extend(getattr(obj, attr, None) for attr in ("inner", "client"))
        return [obj for obj in seen.values() if inspect.iscoroutinefunction(getattr(obj, "close", None))]

    async def close(self) -> None:
        await asyncio.gather(*(o.close() for o in self._closeables()), return_exceptions=True)

    async def __aenter__(self) -> Harness:
        await self.setup()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
