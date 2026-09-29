"""IR metrics with linear gain, matching BEIR / pytrec_eval."""

from __future__ import annotations

import math


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    ideal = _dcg(sorted((float(r) for r in qrels.values() if r > 0), reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return _dcg([float(max(qrels.get(d, 0), 0)) for d in ranked[:k]]) / ideal


def recall_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    relevant = {d for d, r in qrels.items() if r > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def mrr(ranked: list[str], qrels: dict[str, int]) -> float:
    for i, d in enumerate(ranked):
        if qrels.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0
