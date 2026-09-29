"""Local cross-encoder/reranker as a relevance judge. Needs the `local` extra unless a scorer
is injected. Most 1-label rerankers (e.g. bge-reranker) already output [0, 1] via
sentence-transformers; set apply_sigmoid=True for models that return raw logits."""

from __future__ import annotations

import asyncio
import math
from typing import Any, Callable

from agentic_search.core.types import Hit, Query
from agentic_search.models.base import ControllerView, Decision, JudgeResult, Judgment

Scorer = Callable[[list[tuple[str, str]]], Any]


class CrossEncoderJudge:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", *, scorer: Scorer | None = None,
                 snippet_chars: int = 2000, apply_sigmoid: bool = False, id: str | None = None):
        self.model_name = model_name
        self._scorer = scorer
        self.snippet_chars = snippet_chars
        self.apply_sigmoid = apply_sigmoid
        self.id = id or f"cross-encoder:{model_name}"

    def _default_scorer(self) -> Scorer:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(self.model_name)
        return lambda pairs: model.predict(pairs)

    async def judge(self, question: Query, hits: list[Hit]) -> JudgeResult:
        if not hits:
            return JudgeResult(judgments=[])
        if self._scorer is None:
            self._scorer = await asyncio.to_thread(self._default_scorer)
        pairs = [(question.as_text(), h.snippet(self.snippet_chars)) for h in hits]
        scores = await asyncio.to_thread(self._scorer, pairs)
        out = []
        for h, raw in zip(hits, scores):
            s = float(raw)
            p = 1.0 / (1.0 + math.exp(-s)) if self.apply_sigmoid else min(1.0, max(0.0, s))
            out.append(Judgment(key=h.key, p_relevant=p))
        return JudgeResult(judgments=out)

    async def decide(self, view: ControllerView) -> Decision:
        raise NotImplementedError
