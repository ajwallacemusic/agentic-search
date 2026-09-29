"""A Driver built on any LLMClient with tool calling (Claude, GPT, open models, SID-1-style)."""

from __future__ import annotations

from typing import Any

from agentic_search.core.types import Budget, Content, ModelUsage, Query, TextPart
from agentic_search.models.base import (
    DelegateResult, PlannerView, PlanResult, ToolCall, ToolRuntime, ToolSpec,
)
from agentic_search.models.llm import ChatMessage, LLMClient

PLANNER_PROMPT = """You are the planning component of an agentic search system. Your job is to find \
every document in the available data sources that is relevant to the user's question. You do not \
answer the question.

Each turn you see the question, a description of each data source (collections, fields, \
capabilities), a digest of what earlier searches returned and which results were judged relevant, \
and sometimes a directive from the controller (REFINE, BROADEN, SWITCH_SOURCE, CONTINUE).

Respond with tool calls only. Guidelines:
- Issue several diverse searches in parallel each turn (up to {max_calls}): narrow and broad keyword \
searches, synonyms and domain terminology, and vector searches whose hyde_text is a short passage \
written the way a relevant document would be written.
- Use filters when the question implies constraints and the source has matching filterable fields.
- Use only sources, collections and fields that appear in the source descriptions. Call discover \
for more detail on a collection.
- Never repeat a search that already ran. Learn from what was judged relevant and not relevant.
- If a call failed, read the error and correct it.
- If you believe the search is complete, make no tool calls."""

DELEGATE_PROMPT = """You are a search agent. Find every document in the data sources below that is \
relevant to the user's question. You do not answer the question.

Search iteratively: issue several diverse searches in parallel per turn (keyword variants, synonyms, \
vector searches with a hypothetical relevant passage as hyde_text, filtered searches), read the \
results, and refine. Results list documents as `source:doc_id` keys. Use only the sources, \
collections and fields described below. When you have found enough, call finish with ranked_keys: \
the keys of the relevant documents, most relevant first. If told the budget is exhausted, call \
finish immediately.

# Data sources
{context}"""

FINISH_TOOL = ToolSpec(
    name="finish",
    description="End the search and return the relevant document keys, most relevant first.",
    parameters={"type": "object",
                "properties": {"ranked_keys": {"type": "array", "items": {"type": "string"}}},
                "required": ["ranked_keys"]},
)


def render_planner_view(view: PlannerView) -> str:
    parts = [f"# Question\n{view.question.as_text()}"]
    images = view.question.images()
    if images:
        parts.append(f"(The question includes {len(images)} image(s).)")
    parts.append(f"# Data sources\n{view.manifest_summary}")
    parts.append(f"# Turn {view.turn}\n## Results so far\n{view.digest}")
    if view.directive is not None:
        note = f": {view.directive.note}" if view.directive.note else ""
        parts.append(f"## Controller directive: {view.directive.action.value.upper()}{note}")
    parts.append(f"Issue up to {view.max_calls} tool calls now, or none if the search is complete.")
    return "\n\n".join(parts)


def _finish_keys(call: ToolCall) -> list[str]:
    keys = call.arguments.get("ranked_keys", [])
    return [str(k) for k in keys] if isinstance(keys, list) else []


class ToolCallingDriver:
    def __init__(self, client: LLMClient, *, max_calls_per_turn: int = 8,
                 planner_prompt: str | None = None, delegate_prompt: str | None = None,
                 max_tokens: int = 4096):
        self.client = client
        self.id = client.id
        self.supports_images = client.supports_images
        self.max_calls = max_calls_per_turn
        self.planner_prompt = planner_prompt or PLANNER_PROMPT
        self.delegate_prompt = delegate_prompt or DELEGATE_PROMPT
        self.max_tokens = max_tokens

    def _with_images(self, text: str, question: Query) -> str | list[Content]:
        images = question.images() if self.supports_images else []
        return [TextPart(text=text), *images] if images else text

    async def plan(self, view: PlannerView, tools: list[ToolSpec]) -> PlanResult:
        limit = min(view.max_calls, self.max_calls)
        system = self.planner_prompt.format(max_calls=limit)
        content = self._with_images(render_planner_view(view), view.question)
        resp = await self.client.chat(system, [ChatMessage(role="user", content=content)],
                                      tools=tools, max_tokens=self.max_tokens)
        return PlanResult(calls=resp.tool_calls[:limit], usage=resp.usage, note=resp.text or None)

    async def run_delegate(self, question: Query, tools: list[ToolSpec], runtime: ToolRuntime,
                           budget: Budget, context: str = "") -> DelegateResult:
        system = self.delegate_prompt.format(context=context or "(see tool descriptions)")
        opening = f"{context}\n\n# Question\n{question.as_text()}" if context else question.as_text()
        messages = [ChatMessage(role="user", content=self._with_images(opening, question))]
        usage = ModelUsage()
        all_tools = [*tools, FINISH_TOOL]
        for _ in range(budget.max_turns + 1):
            resp = await self.client.chat(system, messages, tools=all_tools, max_tokens=self.max_tokens)
            usage = usage.plus(resp.usage)
            finish = next((c for c in resp.tool_calls if c.name == FINISH_TOOL.name), None)
            if finish is not None:
                return DelegateResult(ranked_keys=_finish_keys(finish), usage=usage,
                                      note=resp.text or None)
            if not resp.tool_calls:
                return DelegateResult(ranked_keys=[], usage=usage, note=resp.text or None)
            calls = resp.tool_calls[: self.max_calls]
            messages.append(ChatMessage(role="assistant", content=resp.text, tool_calls=calls))
            outputs = await runtime.call(calls)
            messages.extend(ChatMessage(role="tool", tool_call_id=c.id, content=o)
                            for c, o in zip(calls, outputs))
        resp = await self.client.chat(system, messages, tools=[FINISH_TOOL],
                                      tool_choice=FINISH_TOOL.name, max_tokens=self.max_tokens)
        usage = usage.plus(resp.usage)
        finish = next((c for c in resp.tool_calls if c.name == FINISH_TOOL.name), None)
        keys: list[Any] = _finish_keys(finish) if finish is not None else []
        return DelegateResult(ranked_keys=keys, usage=usage, note=resp.text or None)
