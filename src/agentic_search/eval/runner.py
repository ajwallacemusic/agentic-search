"""Run a Harness over a labeled dataset and report quality, cost and latency."""

from __future__ import annotations

import asyncio
import time

from pydantic import BaseModel

from agentic_search.core.harness import Harness, Mode
from agentic_search.eval.datasets import BeirDataset
from agentic_search.eval.metrics import mrr, ndcg_at_k, recall_at_k


class QueryRun(BaseModel):
    query_id: str
    ndcg: float
    recall: float
    mrr: float
    cost_usd: float
    seconds: float
    tool_calls: int
    stop_reason: str
    error: str | None = None


class EvalReport(BaseModel):
    name: str
    k: int
    recall_k: int
    runs: list[QueryRun]

    def mean(self, attr: str) -> float:
        values = [float(getattr(r, attr)) for r in self.runs]
        return sum(values) / len(values) if values else 0.0

    def table(self) -> str:
        errors = sum(r.error is not None for r in self.runs)
        return (f"{self.name:<28} nDCG@{self.k}={self.mean('ndcg'):.4f} "
                f"R@{self.recall_k}={self.mean('recall'):.4f} MRR={self.mean('mrr'):.4f} "
                f"cost=${self.mean('cost_usd'):.4f}/q latency={self.mean('seconds'):.2f}s/q "
                f"calls={self.mean('tool_calls'):.1f}/q n={len(self.runs)} errors={errors}")


async def run_eval(harness: Harness, dataset: BeirDataset, *, name: str,
                   query_ids: list[str] | None = None, top_k: int = 100, k: int = 10,
                   recall_k: int = 100, concurrency: int = 4, mode: Mode | None = None) -> EvalReport:
    qids = query_ids or sorted(dataset.queries)
    sem = asyncio.Semaphore(concurrency)

    async def one(qid: str) -> QueryRun:
        async with sem:
            t0 = time.perf_counter()
            try:
                res = await harness.search(dataset.queries[qid], top_k=top_k, mode=mode)
            except Exception as exc:
                return QueryRun(query_id=qid, ndcg=0.0, recall=0.0, mrr=0.0, cost_usd=0.0,
                                seconds=time.perf_counter() - t0, tool_calls=0, stop_reason="error",
                                error=f"{type(exc).__name__}: {exc}")
            ranked = [h.hit.doc_id for h in res.hits]
            qrels = dataset.qrels.get(qid, {})
            return QueryRun(query_id=qid, ndcg=ndcg_at_k(ranked, qrels, k),
                            recall=recall_at_k(ranked, qrels, recall_k), mrr=mrr(ranked, qrels),
                            cost_usd=res.usage.cost_usd, seconds=time.perf_counter() - t0,
                            tool_calls=res.usage.tool_calls, stop_reason=res.stop_reason.value)

    runs = await asyncio.gather(*(one(q) for q in qids))
    return EvalReport(name=name, k=k, recall_k=recall_k, runs=list(runs))
