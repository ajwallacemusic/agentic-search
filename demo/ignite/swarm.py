"""Swarm mode: split a question, divide the budget, run the agents, merge ranked lists (RRF).

The swarm splits one question into independent sub-questions, runs one harness agent per part in
parallel, then merges the agents' ranked documents with reciprocal rank fusion. Design:
docs/superpowers/specs/2026-11-19-ignite-swarm-design.md."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from collections.abc import AsyncIterator
from typing import Any, Literal

from pydantic import BaseModel

from agentic_search.core.harness import Harness
from agentic_search.core.result import RankedHit, SearchResult
from agentic_search.core.secrets import scrub
from agentic_search.core.state import Trace, Usage
from agentic_search.core.types import Budget, Hit, ModelUsage, Query, StopReason
from agentic_search.events import (
    FactNoted,
    ResultsUpdated,
    SearchFailed,
    SearchFinished,
    UsageUpdated,
)
from agentic_search.models.llm import ChatMessage
from agentic_search.server.models import render_event

logger = logging.getLogger(__name__)

SPLIT_SYSTEM = (
    "You split a question into independent sub-questions for a team of search agents. Each agent "
    "searches the sources below on its own, so every sub-question must make sense without the "
    "others. Write 3 to 6 sub-questions that together cover what is needed to answer the "
    "question. Reply with only JSON: {\"sub_questions\": [\"...\", \"...\"]}.")


class SwarmConfig(BaseModel):
    max_agents: int = 4
    agent_max_turns: int = 2
    rrf_k: int = 60


class Split(BaseModel):
    sub_questions: list[str]
    fallback: bool
    usage: ModelUsage


def parse_split(text: str, question: str, max_agents: int) -> tuple[list[str], bool]:
    """(sub-questions, fallback). Fewer than two usable parts means no split: the question
    itself, flagged as a fallback."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    subs: list[str] = []
    if match:
        try:
            raw = json.loads(match.group(0)).get("sub_questions", [])
        except (ValueError, AttributeError):
            raw = []
        if isinstance(raw, list):
            for item in raw:
                s = item.strip() if isinstance(item, str) else ""
                if s and s not in subs:
                    subs.append(s)
    subs = subs[:max_agents]
    if len(subs) < 2:
        return [question], True
    return subs, False


async def split_question(client: Any, question: str, descriptions: dict[str, str],
                         max_agents: int) -> Split:
    sources = "\n".join(f"- {name}: {desc}" for name, desc in descriptions.items())
    prompt = f"Question: {question}\n\nSources:\n{sources}"
    try:
        resp = await client.chat(SPLIT_SYSTEM, [ChatMessage(role="user", content=prompt)],
                                 max_tokens=800)
    except Exception:
        return Split(sub_questions=[question], fallback=True, usage=ModelUsage())
    subs, fallback = parse_split(resp.text, question, max_agents)
    return Split(sub_questions=subs, fallback=fallback, usage=resp.usage)


def agent_budget(total: Budget, n: int, split_cost: float, max_turns: int) -> Budget:
    """One agent's share of the harness budget: tool calls and cost split `n` ways (cost after
    the split call is paid for); time and token limits are kept as is."""
    cost = None if total.max_cost_usd is None else max(0.0, total.max_cost_usd - split_cost) / n
    return Budget(max_turns=max_turns, max_tool_calls=max(1, total.max_tool_calls // n),
                  max_tokens=total.max_tokens, max_cost_usd=cost, max_seconds=total.max_seconds)


def agent_query(question: str, part: str) -> str:
    return f"Main question: {question}. Your part: {part}."


def rrf(ranked: list[list[str]], k: int) -> list[tuple[str, float, list[int]]]:
    """Reciprocal rank fusion: (key, score, indexes of the lists that held it), best first.
    Ties keep first-seen order."""
    scores: dict[str, float] = {}
    sources: dict[str, list[int]] = {}
    for idx, keys in enumerate(ranked):
        for rank, key in enumerate(keys, 1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            sources.setdefault(key, []).append(idx)
    order = sorted(scores, key=lambda key: -scores[key])  # stable: ties keep insertion order
    return [(key, scores[key], sources[key]) for key in order]


AgentStatus = Literal["done", "failed", "cut"]


class _Agent:
    """What one agent has reported so far. `status` stays "cut" until its stream ends."""

    def __init__(self) -> None:
        self.keys: list[str] = []            # ranked: final hits if finished, else last snapshot
        self.info: dict[str, dict[str, Any]] = {}   # key -> source, doc_id, title, snippet
        self.notes: list[str] = []
        self.usage = Usage()
        self.status: AgentStatus = "cut"

    def record(self, event: Any) -> None:
        if isinstance(event, ResultsUpdated):
            self.keys = [h.key for h in event.hits]
            for h in event.hits:
                self.info[h.key] = {"source": h.source, "doc_id": h.doc_id, "title": h.title,
                                    "snippet": h.snippet}
        elif isinstance(event, FactNoted):
            self.notes.append(event.fact)
        elif isinstance(event, UsageUpdated):
            self.usage = event.usage
        elif isinstance(event, SearchFinished):
            self.status, self.usage = "done", event.result.usage
            self.keys = [r.hit.key for r in event.result.hits]
            for r in event.result.hits:
                title = r.hit.metadata.get("title")
                self.info.setdefault(r.hit.key, {
                    "source": r.hit.source, "doc_id": r.hit.doc_id,
                    "title": title if isinstance(title, str) else None,
                    "snippet": r.hit.snippet(300)})
        elif isinstance(event, SearchFailed):
            self.status = "failed"


class Swarm:
    """Split, fan out over one shared harness (each agent with its share of the budget), fuse.

    `stream` yields the swarm wire events; `search` has the `Harness.search` shape the eval
    runners call."""

    def __init__(self, harness: Harness, client: Any, cfg: SwarmConfig,
                 descriptions: dict[str, str]):
        self.harness, self.client, self.cfg, self.descriptions = harness, client, cfg, descriptions

    async def __aenter__(self) -> Swarm:
        await self.harness.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.harness.__aexit__(*exc)

    async def stream(self, question: str, *, snapshot_k: int = 10,
                     include_trace: bool = False) -> AsyncIterator[dict[str, Any]]:
        """`swarm_started`, `swarm_split`, interleaved `agent_event`s, then one `swarm_finished`
        or `swarm_failed`. The run is cut at the harness budget's `max_seconds` (counted from
        the start); agents still running are cancelled ("cut") and their last snapshot kept."""
        t0 = time.monotonic()

        def at_ms() -> float:
            return round((time.monotonic() - t0) * 1000, 1)

        tasks: list[asyncio.Task[None]] = []
        try:
            total = self.harness.budget
            yield {"type": "swarm_started", "question": question,
                   "max_agents": self.cfg.max_agents, "budget": total.model_dump(mode="json"),
                   "at_ms": 0.0}
            split = await split_question(self.client, question, self.descriptions,
                                         self.cfg.max_agents)
            yield {"type": "swarm_split", "sub_questions": split.sub_questions,
                   "fallback": split.fallback, "usage": split.usage.model_dump(mode="json"),
                   "at_ms": at_ms()}

            n = len(split.sub_questions)
            budget = agent_budget(total, n, split.usage.cost_usd, self.cfg.agent_max_turns)
            agents = [_Agent() for _ in range(n)]
            queue: asyncio.Queue[tuple[int, Any]] = asyncio.Queue()
            for i, part in enumerate(split.sub_questions):
                # A fallback agent searches the question itself, unchanged.
                query = question if split.fallback else agent_query(question, part)
                tasks.append(asyncio.create_task(self._run_agent(
                    i, query, budget, snapshot_k, agents[i], queue)))

            deadline = None if total.max_seconds is None else t0 + total.max_seconds
            running = n
            while running:
                timeout = None if deadline is None else deadline - time.monotonic()
                if timeout is not None and timeout <= 0:
                    break
                try:
                    i, event = await asyncio.wait_for(queue.get(), timeout)
                except TimeoutError:
                    break
                if event is None:  # agent i's stream ended
                    running -= 1
                    continue
                yield _agent_event(i, event, include_trace)
            await _cancel(tasks)
            while not queue.empty():  # events queued at the deadline: their hits are merged
                i, event = queue.get_nowait()
                if event is not None:
                    yield _agent_event(i, event, include_trace)
            yield self._finished(split, agents, snapshot_k, time.monotonic() - t0, at_ms())
        except Exception as exc:
            yield {"type": "swarm_failed", "error_type": type(exc).__name__, "message": str(exc),
                   "at_ms": at_ms()}
        finally:
            await _cancel(tasks)

    async def _run_agent(self, idx: int, query: str, budget: Budget, snapshot_k: int,
                         agent: _Agent, queue: asyncio.Queue[tuple[int, Any]]) -> None:
        """Pump one agent's harness events into the shared queue; (idx, None) marks the end.
        If the stream raises, the cause is logged and sent on as a `search_failed`-shaped dict."""
        try:
            stream = self.harness.stream(query, top_k=snapshot_k, budget=budget,
                                         snapshot_k=snapshot_k, include_content=False)
            async with contextlib.aclosing(stream) as events:
                async for event in events:
                    agent.record(event)
                    queue.put_nowait((idx, event))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("swarm agent %d failed", idx, exc_info=True)
            agent.status = "failed"
            queue.put_nowait((idx, {"type": "search_failed", "error_type": type(exc).__name__,
                                    "message": scrub(str(exc))}))
        finally:
            if agent.status == "cut" and not _cancelling():
                agent.status = "failed"  # stream ended without a terminal event
            queue.put_nowait((idx, None))

    def _finished(self, split: Split, agents: list[_Agent], snapshot_k: int, seconds: float,
                  at_ms: float) -> dict[str, Any]:
        info: dict[str, dict[str, Any]] = {}
        for agent in agents:
            for key, data in agent.info.items():
                info.setdefault(key, data)
        fused = rrf([a.keys for a in agents], self.cfg.rrf_k)[:snapshot_k]
        hits = [{"key": key, **info[key], "score": round(score, 6), "agents": idx}
                for key, score, idx in fused]
        usage = {
            "input_tokens": split.usage.input_tokens + sum(a.usage.input_tokens for a in agents),
            "output_tokens": split.usage.output_tokens + sum(a.usage.output_tokens for a in agents),
            "cost_usd": split.usage.cost_usd + sum(a.usage.cost_usd for a in agents),
            "tool_calls": sum(a.usage.tool_calls for a in agents),
            "turns": max((a.usage.turns for a in agents), default=0)}
        return {"type": "swarm_finished", "hits": hits,
                "notes": [note for a in agents for note in a.notes], "usage": usage,
                "seconds": round(seconds, 3), "agent_status": [a.status for a in agents],
                "at_ms": at_ms}

    async def search(self, question: str, *, top_k: int = 20, mode: Any = None,
                     **_: Any) -> SearchResult:
        """One swarm run as a SearchResult (merged hits with their RRF scores), for the eval
        runners. Raises RuntimeError if the swarm failed."""
        last: dict[str, Any] = {}
        async with contextlib.aclosing(self.stream(question, snapshot_k=top_k)) as events:
            async for event in events:
                last = event
        if last.get("type") != "swarm_finished":
            raise RuntimeError(last.get("message", "swarm ended without a result"))
        hits = [RankedHit(hit=Hit(doc_id=h["doc_id"], source=h["source"]), score=h["score"])
                for h in last["hits"]]
        return SearchResult(question=Query.of(question), hits=hits,
                            stop_reason=StopReason.DELEGATE_DONE, usage=Usage(**last["usage"]),
                            trace=Trace(), mode="swarm")


def _agent_event(idx: int, event: Any, include_trace: bool) -> dict[str, Any]:
    """Wrap a library event (or an already-shaped dict) as a swarm `agent_event`."""
    if not isinstance(event, dict):
        event = json.loads(render_event(event, include_content=False,
                                        include_trace=include_trace))
    return {"type": "agent_event", "agent": idx, "event": event}


def _cancelling() -> bool:
    task = asyncio.current_task()
    return bool(task and task.cancelling())


async def _cancel(tasks: list[asyncio.Task[None]]) -> None:
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
