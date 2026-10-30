"""Eval helpers shared by public/eval_public.py: a no-model BM25 baseline, per-group runs of the
library's eval runner, and markdown tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agentic_search.backends.files import FilesBackend
from agentic_search.core.harness import Harness
from agentic_search.core.types import Document, Query
from agentic_search.eval.datasets import BeirDataset
from agentic_search.eval.runner import EvalReport, run_eval
from agentic_search.models.base import PlannerView, PlanResult, ToolCall, ToolSpec


class AllSourcesBM25Driver:
    """Turn 0: one BM25 search of the question text per source. Later turns: nothing."""

    id = "bm25-all-sources"
    supports_images = False

    def __init__(self, sources: list[str], limit: int = 20):
        self.sources = sources
        self.limit = limit

    async def plan(self, view: PlannerView, tools: list[ToolSpec]) -> PlanResult:
        if view.turn > 0:
            return PlanResult(calls=[])
        text = view.question.as_text()
        return PlanResult(calls=[ToolCall(id=f"bm25-{s}", name="lexical_search",
                                          arguments={"source": s, "text": text,
                                                     "limit": self.limit})
                                 for s in self.sources])

    async def run_delegate(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def bm25_harness_for(docs: dict[str, list[Document]], descriptions: dict[str, str]) -> Harness:
    backends = [FilesBackend.from_documents(s, d, collection=s, description=descriptions[s])
                for s, d in docs.items()]
    return Harness(backends, AllSourcesBM25Driver(list(docs)), mode="retrieval")


class TurnRecorder:
    """Passes `search` through to a harness and records each result's turn count, which the
    eval runner's per-query rows do not keep."""

    def __init__(self, harness: Harness):
        self.harness = harness
        self.turns: dict[str, int] = {}
        self.top10: dict[str, list[str]] = {}

    async def search(self, question: str | Query, **kwargs: Any) -> Any:
        result = await self.harness.search(question, **kwargs)
        self.turns[str(question)] = result.usage.turns
        self.top10[str(question)] = [h.hit.doc_id for h in result.hits[:10]]
        return result


@dataclass
class KindReport:
    report: EvalReport
    turns: list[int] = field(default_factory=list)
    precision: list[float] = field(default_factory=list)  # P@10 per run; an errored run is 0


def _precision_at_10(top10: list[str], qrels: dict[str, int]) -> float:
    return sum(qrels.get(doc_id, 0) > 0 for doc_id in top10[:10]) / 10


async def evaluate_groups(modes: list[str], harness_for: Callable[[str], Harness],
                          dataset: BeirDataset, groups: dict[str, list[str]], *,
                          name: str) -> dict[tuple[str, str], KindReport]:
    """Run each mode over each group of query ids; a fresh harness per mode."""
    out: dict[tuple[str, str], KindReport] = {}
    for mode in modes:
        harness = harness_for(mode)
        run_mode = "retrieval" if mode == "bm25" else mode
        async with harness:
            for group, qids in groups.items():
                recorder = TurnRecorder(harness)
                report = await run_eval(recorder, dataset,  # type: ignore[arg-type]
                                        name=f"{name}/{mode}/{group}", query_ids=qids, k=10,
                                        recall_k=10, concurrency=2,
                                        mode=run_mode)  # type: ignore[arg-type]
                turns = [recorder.turns.get(dataset.queries[r.query_id], 0) for r in report.runs]
                precision = [_precision_at_10(recorder.top10.get(dataset.queries[r.query_id], []),
                                              dataset.qrels.get(r.query_id, {}))
                             for r in report.runs]
                out[(mode, group)] = KindReport(report=report, turns=turns, precision=precision)
    return out


def results_table(reports: dict[tuple[str, str], KindReport], groups: dict[str, str]) -> list[str]:
    """Markdown lines: one table per group (key -> heading), one row per mode."""
    lines: list[str] = []
    for group, title in groups.items():
        lines += [f"## {title} questions", "",
                  "| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | "
                  "avg cost | avg seconds | n | errors |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for (mode, g), kr in reports.items():
            if g != group:
                continue
            r = kr.report
            turns = sum(kr.turns) / len(kr.turns) if kr.turns else 0.0
            errors = sum(run.error is not None for run in r.runs)
            p10 = sum(kr.precision) / len(kr.precision) if kr.precision else 0.0
            lines.append(f"| {mode} | {r.mean('ndcg'):.3f} | {p10:.3f} | {r.mean('recall'):.3f} | "
                         f"{r.mean('mrr'):.3f} | {turns:.1f} | {r.mean('tool_calls'):.1f} | "
                         f"${r.mean('cost_usd'):.4f} | {r.mean('seconds'):.1f} | {len(r.runs)} | "
                         f"{errors} |")
        lines.append("")
    return lines
