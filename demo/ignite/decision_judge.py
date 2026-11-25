"""A judge backed by Gemini Decision 1 (the Gemini Decisions API), for the Ignite demo.

Decision 1 does not write text. It takes one shared input and up to 1,000 closed questions and
returns a probability per question, so one call grades a whole batch: one yes/no ("noul")
question per hit, "is document [i] relevant to the question?". Output tokens are not billed.

On 2026-11-25 the model is only on the Gemini Developer API (AI Studio key), not on Vertex
(404 in ajw-sandbox-497707). The key comes from `GEMINI_API_KEY`, else the macOS Keychain item
`gemini-api-key` (`security add-generic-password -a "$USER" -s gemini-api-key -w`).
"""

from __future__ import annotations

import os
import subprocess

import httpx

from agentic_search.core.types import Hit, ModelUsage, Query
from agentic_search.models.base import ControllerView, Decision, JudgeResult, Judgment

URL = "https://generativelanguage.googleapis.com/v1beta/decisions"
QUESTION = ("Is document [{i}] relevant to the question: does it contain information that helps "
            "answer it, or is it one of the things the question asks for?")


def api_key(env: str = "GEMINI_API_KEY", keychain_item: str = "gemini-api-key") -> str:
    key = os.environ.get(env)
    if key:
        return key
    try:
        return subprocess.check_output(["security", "find-generic-password", "-s", keychain_item,
                                        "-w"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"no Gemini API key: set {env} or add Keychain item {keychain_item}") \
            from exc


class DecisionJudge:
    def __init__(self, *, model: str = "gemini-decision-1", key: str | None = None,
                 price_per_mtok_in: float = 0.10, snippet_chars: int = 1200,
                 request_timeout_s: float = 30.0, max_retries: int = 1,
                 client: httpx.AsyncClient | None = None):
        self.model = model
        self.id = f"decision-judge:{model}"
        self._key = key
        self.price_in = price_per_mtok_in
        self.snippet_chars = snippet_chars
        self.timeout = request_timeout_s
        self.max_retries = max_retries
        self._client = client

    def body(self, question: Query, hits: list[Hit]) -> dict:
        docs = "\n\n".join(f"[{i}] {h.key}\n{h.snippet(self.snippet_chars) or '(no content)'}"
                           for i, h in enumerate(hits))
        return {"model": self.model,
                "input": f"Question: {question.as_text()}\n\nDocuments:\n{docs}",
                "questions": {f"d{i}": {"type": "noul", "instructions": QUESTION.format(i=i)}
                              for i in range(len(hits))}}

    def parse(self, data: dict, hits: list[Hit]) -> JudgeResult:
        answers = data.get("answers", {})
        out = []
        for i, h in enumerate(hits):
            a = answers.get(f"d{i}")
            if a is None or a.get("noul") is None:
                continue
            out.append(Judgment(key=h.key, p_relevant=min(1.0, max(0.0, float(a["noul"])))))
        n_in = int(data.get("usage", {}).get("total_input_tokens", 0))
        return JudgeResult(judgments=out, usage=ModelUsage(
            input_tokens=n_in, output_tokens=0, cost_usd=n_in * self.price_in / 1e6))

    async def judge(self, question: Query, hits: list[Hit]) -> JudgeResult:
        if not hits:
            return JudgeResult(judgments=[])
        if self._key is None:
            self._key = api_key()
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        try:
            for attempt in range(self.max_retries + 1):
                try:
                    r = await client.post(URL, json=self.body(question, hits),
                                          headers={"x-goog-api-key": self._key})
                    if r.status_code >= 500 or r.status_code == 429:
                        r.raise_for_status()
                    if r.status_code >= 400:  # a bad request will not get better on retry
                        raise ValueError(f"decisions API {r.status_code}: {r.text[:200]}")
                    return self.parse(r.json(), hits)
                except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError):
                    if attempt == self.max_retries:
                        raise
            raise AssertionError("unreachable")
        finally:
            if self._client is None:
                await client.aclose()

    async def decide(self, view: ControllerView) -> Decision:
        raise NotImplementedError
