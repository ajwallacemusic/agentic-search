"""Swarm mode, pure parts: split a question, divide the budget, merge ranked lists (RRF).

The swarm splits one question into independent sub-questions, runs one harness agent per part in
parallel, then merges the agents' ranked documents with reciprocal rank fusion."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel

from agentic_search.core.types import Budget, ModelUsage
from agentic_search.models.llm import ChatMessage

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
