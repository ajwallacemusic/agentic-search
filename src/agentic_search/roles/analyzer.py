"""Judges new candidates, blends scores, and writes the digest the planner reads next turn."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass

from agentic_search.core.hooks import Hooks, SourcePolicy
from agentic_search.core.state import Candidate, SearchState
from agentic_search.core.types import ModelUsage
from agentic_search.models.base import Decider, ToolCall
from agentic_search.roles.executor import ExecResult


@dataclass
class AnalysisResult:
    n_new: int
    n_new_relevant: int | None
    digest: str
    usage: ModelUsage


class Analyzer:
    def __init__(self, decider: Decider | None, *, hooks: Hooks | None = None,
                 policy: SourcePolicy | None = None, relevant_threshold: float = 0.5,
                 judge_weight: float = 0.9, batch_size: int = 16, timeout: float = 60.0,
                 digest_hits: int = 8, snippet_chars: int = 240):
        self.decider = decider
        self.hooks = hooks or Hooks()
        self.policy = policy or SourcePolicy()
        self.relevant_threshold = relevant_threshold
        self.judge_weight = judge_weight
        self.batch_size = batch_size
        self.timeout = timeout
        self.digest_hits = digest_hits
        self.snippet_chars = snippet_chars
        self._can_judge = decider is not None

    def is_relevant(self, cand: Candidate) -> bool:
        return cand.judged and cand.p_relevant is not None and cand.p_relevant >= self.relevant_threshold

    async def judge_keys(self, state: SearchState, keys: list[str]) -> ModelUsage:
        usage = ModelUsage()
        pending = [k for k in keys if not state.pool[k].judged]
        if not pending:
            return usage
        if not self._can_judge or self.decider is None:
            for k in pending:
                state.pool[k].unjudged_reason = "no judge configured"
            return usage
        decider = self.decider
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start:start + self.batch_size]
            hits = self.policy.redact([state.pool[k].hit for k in batch], decider.id)
            hits = await self.hooks.before_model_call(decider.id, hits)
            t0 = time.perf_counter()
            try:
                result = await asyncio.wait_for(decider.judge(state.question, hits), self.timeout)
            except NotImplementedError:
                self._can_judge = False
                for k in pending[start:]:
                    state.pool[k].unjudged_reason = f"{decider.id} does not judge"
                return usage
            except Exception as exc:
                for k in batch:
                    state.pool[k].unjudged_reason = f"judge failed: {type(exc).__name__}"
                state.trace.add("judge_error", state.turn, decider=decider.id, n=len(batch),
                                error=f"{type(exc).__name__}: {str(exc)[:300]}")
                continue
            usage = usage.plus(result.usage)
            by_key = {j.key: j for j in result.judgments}
            n_relevant = 0
            for k in batch:
                cand, j = state.pool[k], by_key.get(k)
                if j is None:
                    cand.unjudged_reason = "no judgment returned"
                    continue
                cand.p_relevant, cand.rationale = j.p_relevant, j.rationale
                cand.judged, cand.unjudged_reason = True, None
                n_relevant += self.is_relevant(cand)
            state.trace.add("judge", state.turn, duration_ms=(time.perf_counter() - t0) * 1000,
                            decider=decider.id, n=len(batch), n_relevant=n_relevant)
        return usage

    async def analyze(self, state: SearchState, calls: list[ToolCall], result: ExecResult, *,
                      digest_model_id: str) -> AnalysisResult:
        usage = await self.judge_keys(state, result.new_keys)
        judged_new = [state.pool[k] for k in result.new_keys if state.pool[k].judged]
        n_rel = sum(self.is_relevant(c) for c in judged_new) if judged_new else None
        digest = self._digest(state, calls, result, digest_model_id)
        return AnalysisResult(n_new=len(result.new_keys), n_new_relevant=n_rel, digest=digest,
                              usage=usage)

    def _digest(self, state: SearchState, calls: list[ToolCall], result: ExecResult,
                model_id: str) -> str:
        pool = state.pool
        new = [pool[k] for k in result.new_keys]
        relevant = [c for c in new if self.is_relevant(c)]
        irrelevant = [c for c in new if c.judged and not self.is_relevant(c)]
        header = f"Turn {state.turn}: {len(calls)} calls, {len(new)} new candidates"
        if any(c.judged for c in new):
            header += f", {len(relevant)} judged relevant"
        lines = [f"{header}. Pool size: {len(pool)}.", "Per call:"]
        errors = {e.call_id: e for e in result.errors}
        new_set = set(result.new_keys)
        for c in calls:
            args = json.dumps(c.arguments, default=str)[:200]
            if c.id in errors:
                lines.append(f"- {c.name} {args} -> {errors[c.id].render()}")
                continue
            keys = result.hits_per_call.get(c.id)
            if keys is None:
                lines.append(f"- {c.name} {args} -> ok")
                continue
            line = f"- {c.name} {args} -> {len(keys)} hits, {len(new_set.intersection(keys))} new"
            if self._can_judge:
                line += f", {sum(self.is_relevant(pool[k]) for k in keys)} relevant"
            lines.append(line)
        if relevant:
            lines.append("New relevant:")
            relevant.sort(key=lambda c: -(c.p_relevant or 0.0))
            lines.extend(self._line(c, model_id) for c in relevant[: self.digest_hits])
        if irrelevant:
            lines.append("Judged not relevant (examples):")
            lines.extend(self._line(c, model_id) for c in irrelevant[:3])
        if new and not relevant and not irrelevant:
            lines.append("New candidates (not judged):")
            lines.extend(self._line(c, model_id) for c in new[: self.digest_hits])
        top = pool.ranked(self.judge_weight)[:5]
        if top:
            lines.append("Current top results: " + ", ".join(f"{c.hit.key} ({s:.2f})" for c, s in top))
        return "\n".join(lines)

    def _line(self, cand: Candidate, model_id: str) -> str:
        h = cand.hit
        if self.policy.allows(h.source, model_id):
            body = h.snippet(self.snippet_chars)
        else:
            body = "(content withheld by source policy)"
        p = f" p={cand.p_relevant:.2f}" if cand.judged and cand.p_relevant is not None else ""
        why = f" ({cand.rationale})" if cand.rationale else ""
        return f"- {h.key}{p}: {body}{why}"
