"""TypeSafe System One models (Jev, …) as deciders.

judge(): one request per batch of hits. The shared state holds the query and the documents; each
document gets its own Noul ("does it help answer the query?"), so the batch is judged in parallel
and every hit gets a calibrated probability. decide(): one Choice over the controller actions.

HTTP contract: POST https://api.typesafe.ai/v1/systemone with `Authorization: Bearer <key>`,
body {state, model, questions: {id: {type, instructions, criteria}}}; see docs.typesafe.ai/api."""

from __future__ import annotations

import asyncio
import os
import random
from typing import Any

import httpx

from agentic_search.core.secrets import register_secret, scrub
from agentic_search.core.types import Hit, ModelUsage, Query
from agentic_search.models.base import (
    Action,
    ControllerView,
    Decision,
    JudgeResult,
    Judgment,
)
from agentic_search.models.llm import usage_with_cost

API_URL = "https://api.typesafe.ai/v1/systemone"
RETRY_STATUSES = {408, 429, 500, 502, 503, 504, 529}

RELEVANCE_CRITERIA = {
    "true": "The document contains information that helps answer the query: it states facts, "
            "findings or instructions the query is asking for.",
    "false": "The document is off-topic, or only shares vocabulary or a general subject with the "
             "query without supplying what the query asks for.",
}
ACTION_CRITERIA = {
    Action.STOP.value: "Enough relevant documents have been found, or more searching is unlikely "
                       "to find more.",
    Action.REFINE.value: "Searches return too much noise; make the queries more specific.",
    Action.BROADEN.value: "Searches return too little; widen or rephrase the queries.",
    Action.SWITCH_SOURCE.value: "The current sources look exhausted; try other sources.",
    Action.CONTINUE.value: "Searches are productive; keep exploring in the same direction.",
}


class TypeSafeError(Exception):
    pass


class TypeSafeDecider:
    def __init__(self, model: str = "jev-latest", *, api_key: str | None = None,
                 url: str = API_URL, client: httpx.AsyncClient | None = None, id: str | None = None,
                 batch_size: int = 16, snippet_chars: int = 2000, timeout_s: float = 60.0,
                 max_retries: int = 4, backoff_s: float = 0.5,
                 price_per_mtok: tuple[float, float] | None = None):
        self.model = model
        self.id = id or f"typesafe:{model}"
        self._api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if self._api_key:
            register_secret(self._api_key)
        self.url = url
        self.batch_size = max(1, batch_size)
        self.snippet_chars = snippet_chars
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.price = price_per_mtok
        self._client = client
        self._owns_client = client is None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client

    async def _ask(self, state: Any, questions: dict[str, Any]) -> tuple[dict[str, Any], ModelUsage]:
        if not self._api_key:
            raise TypeSafeError("TypeSafe needs an API key (api_key= or TYPESAFE_API_KEY)")
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = {"state": state, "model": self.model, "questions": questions}
        last = ""
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._get_client().post(self.url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code < 300:
                    try:
                        body = resp.json()
                        usage = body.get("usage") or {}
                        return body.get("answers") or {}, usage_with_cost(
                            int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)), self.price)
                    except (ValueError, AttributeError, TypeError) as exc:
                        raise TypeSafeError(scrub(f"{self.id}: malformed response: {type(exc).__name__}")) from exc
                last = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if resp.status_code not in RETRY_STATUSES:
                    break
            if attempt < self.max_retries:
                await asyncio.sleep(self.backoff_s * (2 ** attempt) * (1 + random.random()))
        raise TypeSafeError(scrub(f"{self.id}: {last}"))

    async def judge(self, question: Query, hits: list[Hit]) -> JudgeResult:
        judgments: list[Judgment] = []
        usage = ModelUsage()
        for start in range(0, len(hits), self.batch_size):
            batch = hits[start:start + self.batch_size]
            state = {"query": question.as_text(),
                     "documents": [{"text": h.snippet(self.snippet_chars) or "(no content)"}
                                   for h in batch]}
            questions = {
                f"d{i}": {
                    "type": "noul",
                    "instructions": f"Does the document `documents[{i}].text` help answer the query "
                                    f"`query`?",
                    "criteria": RELEVANCE_CRITERIA,
                } for i in range(len(batch))
            }
            try:
                answers, u = await self._ask(state, questions)
            except TypeSafeError:
                # If at least one batch succeeded, return partial results; otherwise re-raise
                if judgments:
                    return JudgeResult(judgments=judgments, usage=usage)
                raise
            usage = usage.plus(u)
            for i, h in enumerate(batch):
                p = (answers.get(f"d{i}") or {}).get("noul")
                if isinstance(p, (int, float)):
                    judgments.append(Judgment(key=h.key, p_relevant=min(1.0, max(0.0, float(p)))))
        return JudgeResult(judgments=judgments, usage=usage)

    async def decide(self, view: ControllerView) -> Decision:
        state = {
            "question": view.question.as_text(),
            "turns": [t.model_dump() for t in view.history],
            "total_relevant": view.total_relevant,
            "budget_remaining": view.budget_remaining,
            "latest_results": view.digest,
        }
        questions = {"action": {
            "type": "choice",
            "instructions": "An agentic search loop is looking for every document relevant to "
                            "`question`. Given the per-turn statistics in `turns`, the relevant "
                            "documents found so far (`total_relevant`), `budget_remaining` and "
                            "`latest_results`, what should the search do next?",
            "criteria": ACTION_CRITERIA,
        }}
        answers, usage = await self._ask(state, questions)
        answer = answers.get("action") or {}
        try:
            action = Action(answer["choice"])
        except (KeyError, ValueError) as exc:
            raise TypeSafeError(f"{self.id}: no usable choice in response") from exc
        confidence = answer.get("confidence")
        confidence_val = 1.0
        if confidence is not None and isinstance(confidence, (int, float)):
            confidence_val = min(1.0, max(0.0, float(confidence)))
        return Decision(action=action, usage=usage, note="typesafe", confidence=confidence_val)

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None
