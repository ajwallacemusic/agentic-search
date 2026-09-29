"""Decider backed by any LLMClient: relevance grading and stop/continue decisions."""

from __future__ import annotations

import json

from agentic_search.core.types import Content, Hit, ImagePart, Query, TextPart
from agentic_search.models.base import (
    Action,
    ControllerView,
    Decision,
    JudgeResult,
    Judgment,
    ToolSpec,
)
from agentic_search.models.llm import ChatMessage, LLMClient

JUDGE_SYSTEM = """You grade search results for relevance to a question. For each numbered document, \
estimate the probability (0 to 1) that it is relevant, meaning it contains information that helps \
answer the question. Be calibrated: 0.9+ only for clearly relevant documents, under 0.2 for \
off-topic ones. Give a rationale of at most 20 words. Call submit_judgments with one entry per \
document."""

DECIDE_SYSTEM = """You control an agentic search loop. Given the question, per-turn statistics, the \
remaining budget and a digest of the latest results, choose the next action: STOP if enough relevant \
documents were found or more searching is unlikely to help; REFINE to make queries more precise; \
BROADEN to widen them; SWITCH_SOURCE to try other sources; CONTINUE otherwise. Call submit_decision."""

JUDGE_TOOL = ToolSpec(name="submit_judgments", description="Submit one judgment per document.",
                      parameters={"type": "object", "properties": {"judgments": {
                          "type": "array", "items": {"type": "object", "properties": {
                              "index": {"type": "integer"},
                              "p_relevant": {"type": "number", "minimum": 0, "maximum": 1},
                              "rationale": {"type": "string"}},
                              "required": ["index", "p_relevant"]}}},
                          "required": ["judgments"]})

DECISION_TOOL = ToolSpec(name="submit_decision", description="Submit the next action.",
                         parameters={"type": "object", "properties": {
                             "action": {"type": "string", "enum": [a.value for a in Action]},
                             "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                             "note": {"type": "string"}},
                             "required": ["action"]})


def _clamp(x: float) -> float:
    return min(1.0, max(0.0, x))


class LLMJudge:
    def __init__(self, client: LLMClient, *, snippet_chars: int = 1200, id: str | None = None,
                 max_tokens: int = 2048):
        self.client = client
        self.snippet_chars = snippet_chars
        self.max_tokens = max_tokens
        self.id = id or f"llm-judge:{client.id}"

    async def judge(self, question: Query, hits: list[Hit]) -> JudgeResult:
        if not hits:
            return JudgeResult(judgments=[])
        docs = "\n\n".join(f"[{i}] {h.key}\n{h.snippet(self.snippet_chars) or '(no content)'}"
                           for i, h in enumerate(hits))
        content: list[Content] = [TextPart(text=f"Question: {question.as_text()}\n\nDocuments:\n{docs}")]
        if self.client.supports_images:
            content += question.images()
            for i, h in enumerate(hits):
                for part in h.content:
                    if isinstance(part, ImagePart):
                        content += [TextPart(text=f"Image for document [{i}]:"), part]
        resp = await self.client.chat(JUDGE_SYSTEM, [ChatMessage(role="user", content=content)],
                                      tools=[JUDGE_TOOL], tool_choice=JUDGE_TOOL.name,
                                      max_tokens=self.max_tokens)
        call = next((c for c in resp.tool_calls if c.name == JUDGE_TOOL.name), None)
        if call is None:
            raise ValueError("judge model did not call submit_judgments")
        out: list[Judgment] = []
        for item in call.arguments.get("judgments", []):
            try:
                idx, p = int(item["index"]), float(item["p_relevant"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= idx < len(hits):
                why = item.get("rationale")
                out.append(Judgment(key=hits[idx].key, p_relevant=_clamp(p),
                                    rationale=str(why) if why is not None else None))
        return JudgeResult(judgments=out, usage=resp.usage)

    async def decide(self, view: ControllerView) -> Decision:
        history = "\n".join(f"turn {t.turn}: {t.n_calls} calls, {t.n_errors} errors, {t.n_new} new, "
                            f"{t.n_new_relevant} new relevant" for t in view.history)
        text = (f"Question: {view.question.as_text()}\n\nTurns so far:\n{history}\n\n"
                f"Total judged relevant: {view.total_relevant}\n"
                f"Budget remaining: {json.dumps(view.budget_remaining)}\n\n"
                f"Latest digest:\n{view.digest}")
        resp = await self.client.chat(DECIDE_SYSTEM, [ChatMessage(role="user", content=text)],
                                      tools=[DECISION_TOOL], tool_choice=DECISION_TOOL.name,
                                      max_tokens=self.max_tokens)
        call = next((c for c in resp.tool_calls if c.name == DECISION_TOOL.name), None)
        if call is None:
            raise ValueError("decider model did not call submit_decision")
        return Decision(action=Action(str(call.arguments["action"]).lower()),
                        confidence=_clamp(float(call.arguments.get("confidence", 1.0))),
                        note=call.arguments.get("note"), usage=resp.usage)
