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
from demo.ignite.swarm import Swarm


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

    def __init__(self, harness: Harness | Swarm):
        self.harness = harness
        self.turns: dict[str, int] = {}
        self.top10: dict[str, list[str]] = {}
        self.ranked: dict[str, list[str]] = {}  # first 30 doc ids (part scores)

    async def search(self, question: str | Query, **kwargs: Any) -> Any:
        result = await self.harness.search(question, **kwargs)
        self.turns[str(question)] = result.usage.turns
        self.top10[str(question)] = [h.hit.doc_id for h in result.hits[:10]]
        self.ranked[str(question)] = [h.hit.doc_id for h in result.hits[:30]]
        return result


@dataclass
class KindReport:
    report: EvalReport
    turns: list[int] = field(default_factory=list)
    precision: list[float] = field(default_factory=list)  # P@10 per run; an errored run is 0
    coverage: list[float] = field(default_factory=list)  # part coverage@10 per run (wands-multi)
    part_recall: list[float] = field(default_factory=list)  # part recall@30 per run (wands-multi)


def _precision_at_10(top10: list[str], qrels: dict[str, int]) -> float:
    return sum(qrels.get(doc_id, 0) > 0 for doc_id in top10[:10]) / 10


def part_coverage(ranked: list[str], parts: dict[str, list[str]], k: int = 10) -> float:
    """Share of parts with at least one of their Exact products in the top `k`."""
    if not parts:
        return 0.0
    top = set(ranked[:k])
    return sum(bool(top & set(gold)) for gold in parts.values()) / len(parts)


def part_recall(ranked: list[str], parts: dict[str, list[str]], k: int = 30) -> float:
    """Mean over parts of found ÷ min(Exact count, k // number of parts), each capped at 1, so
    a part with hundreds of Exact products does not swamp a part with three."""
    if not parts:
        return 0.0
    top = set(ranked[:k])
    share = max(1, k // len(parts))
    scores = [min(1.0, len(top & set(gold)) / min(len(gold), share)) if gold else 0.0
              for gold in parts.values()]
    return sum(scores) / len(scores)


async def evaluate_groups(modes: list[str], harness_for: Callable[[str], Harness | Swarm],
                          dataset: BeirDataset, groups: dict[str, list[str]], *,
                          name: str,
                          parts: dict[str, dict[str, list[str]]] | None = None,
                          ) -> dict[tuple[str, str], KindReport]:
    """Run each mode over each group of query ids; a fresh harness per mode. The swarm runs one
    query at a time: its agents already run in parallel, and 8 at once overload the shared model.
    With `parts` (question id -> part id -> Exact ids), also score part coverage@10 and part
    recall@30 per run; an errored run scores 0."""
    out: dict[tuple[str, str], KindReport] = {}
    for mode in modes:
        harness = harness_for(mode)
        run_mode = "retrieval" if mode == "bm25" else mode
        async with harness:
            for group, qids in groups.items():
                recorder = TurnRecorder(harness)
                report = await run_eval(recorder, dataset,  # type: ignore[arg-type]
                                        name=f"{name}/{mode}/{group}", query_ids=qids, k=10,
                                        recall_k=10, concurrency=1 if mode.startswith("swarm") else 2,
                                        mode=run_mode)  # type: ignore[arg-type]
                turns = [recorder.turns.get(dataset.queries[r.query_id], 0) for r in report.runs]
                precision = [_precision_at_10(recorder.top10.get(dataset.queries[r.query_id], []),
                                              dataset.qrels.get(r.query_id, {}))
                             for r in report.runs]
                kr = KindReport(report=report, turns=turns, precision=precision)
                if parts is not None:
                    for r in report.runs:
                        ranked = ([] if r.error is not None else
                                  recorder.ranked.get(dataset.queries[r.query_id], []))
                        gold = parts.get(r.query_id, {})
                        kr.coverage.append(part_coverage(ranked, gold))
                        kr.part_recall.append(part_recall(ranked, gold))
                out[(mode, group)] = kr
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


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def parts_table(reports: dict[tuple[str, str], KindReport], group: str = "all") -> list[str]:
    """Markdown lines for the wands-multi scores, one row per mode."""
    lines = ["| mode | part coverage@10 | part recall@30 | nDCG@10 | avg cost | avg seconds | "
             "avg tool calls | n | errors |",
             "|---|---|---|---|---|---|---|---|---|"]
    for (mode, g), kr in reports.items():
        if g != group:
            continue
        r = kr.report
        errors = sum(run.error is not None for run in r.runs)
        lines.append(f"| {mode} | {_mean(kr.coverage):.3f} | {_mean(kr.part_recall):.3f} | "
                     f"{r.mean('ndcg') if r.runs else 0.0:.3f} | "
                     f"${r.mean('cost_usd') if r.runs else 0.0:.4f} | "
                     f"{r.mean('seconds') if r.runs else 0.0:.1f} | "
                     f"{r.mean('tool_calls') if r.runs else 0.0:.1f} | {len(r.runs)} | {errors} |")
    lines.append("")
    return lines


_EPS = 1e-9


def verdict(reports: dict[tuple[str, str], KindReport], group: str = "all") -> tuple[str, str]:
    """The pass mark set before the run (design 2026-11-23), checked in order:
    works / spend / unclear / no. gap = swarm coverage - harness coverage."""
    def scores(mode: str) -> tuple[float, float] | None:
        kr = reports.get((mode, group))
        return None if kr is None else (_mean(kr.coverage), _mean(kr.part_recall))

    h, s, s4 = scores("harness"), scores("swarm"), scores("swarm-4x")
    if h is None or s is None:
        raise ValueError("verdict needs harness and swarm rows")

    def passes(x: tuple[float, float]) -> bool:
        return x[0] - h[0] >= 0.10 - _EPS and x[1] >= h[1] - _EPS

    gap = s[0] - h[0]
    nums = (f"swarm part coverage@10 {s[0]:.3f} vs harness {h[0]:.3f} (gap {gap:+.3f}); "
            f"part recall@30 {s[1]:.3f} vs {h[1]:.3f}")
    if s4 is not None:
        nums += f"; swarm-4x {s4[0]:.3f} / {s4[1]:.3f}"
    if passes(s):
        return "works", f"The swarm works here: {nums}."
    if s4 is not None and passes(s4):
        return "spend", f"Only extra spend works: the fair-budget swarm misses the bar, the 4x one passes. {nums}."
    if gap >= 0.05 - _EPS:
        return "unclear", f"Unclear: the swarm is ahead but misses the bar. {nums}."
    return "no", f"The swarm does not work here: {nums}."
