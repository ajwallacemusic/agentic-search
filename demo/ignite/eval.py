"""Score bm25, retrieval and harness modes on the Ignite questions; write results.md.

Run: uv run python -m demo.ignite.eval --modes bm25,retrieval,harness

Uses the library's eval runner on the BEIR export, with the demo's five-source harness (the
generic scripts/eval_beir.py builds a single-source corpus and reports recall@100).
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agentic_search.backends.files import FilesBackend
from agentic_search.core.harness import Harness
from agentic_search.core.types import Document, Query
from agentic_search.eval.datasets import BeirDataset, load_beir
from agentic_search.eval.runner import EvalReport, run_eval
from agentic_search.models.base import PlannerView, PlanResult, ToolCall, ToolSpec
from demo.ignite.build import SOURCE_DESCRIPTIONS, build_demo_harness, load_config
from demo.ignite.corpus import DATA_DIR, SOURCES, load_documents, load_questions
from demo.ignite.export_beir import BEIR_DIR, export_beir

RESULTS_PATH = Path(__file__).parent / "results.md"
KINDS = {"single_hop": "Single-hop", "multi_hop": "Multi-hop"}


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


def bm25_harness(data_dir: Path = DATA_DIR) -> Harness:
    """Plain BM25 per Ignite source, fused; no model calls."""
    docs = load_documents(data_dir)
    return bm25_harness_for({s: docs[s] for s in SOURCES}, SOURCE_DESCRIPTIONS)


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

    async def search(self, question: str | Query, **kwargs: Any) -> Any:
        result = await self.harness.search(question, **kwargs)
        self.turns[str(question)] = result.usage.turns
        return result


@dataclass
class KindReport:
    report: EvalReport
    turns: list[int] = field(default_factory=list)


async def evaluate(modes: list[str], harness_for: Callable[[str], Harness], *,
                   beir_dir: Path = BEIR_DIR) -> dict[tuple[str, str], KindReport]:
    questions = load_questions()
    groups = {kind: [q.id for q in questions.values() if q.kind == kind] for kind in KINDS}
    return await evaluate_groups(modes, harness_for, load_beir(export_beir(beir_dir)), groups,
                                 name="ignite")


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
                out[(mode, group)] = KindReport(report=report, turns=turns)
    return out


def results_table(reports: dict[tuple[str, str], KindReport], groups: dict[str, str]) -> list[str]:
    """Markdown lines: one table per group (key -> heading), one row per mode."""
    lines: list[str] = []
    for group, title in groups.items():
        lines += [f"## {title} questions", "",
                  "| mode | nDCG@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | "
                  "avg seconds | n | errors |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for (mode, g), kr in reports.items():
            if g != group:
                continue
            r = kr.report
            turns = sum(kr.turns) / len(kr.turns) if kr.turns else 0.0
            errors = sum(run.error is not None for run in r.runs)
            lines.append(f"| {mode} | {r.mean('ndcg'):.3f} | {r.mean('recall'):.3f} | "
                         f"{r.mean('mrr'):.3f} | {turns:.1f} | {r.mean('tool_calls'):.1f} | "
                         f"${r.mean('cost_usd'):.4f} | {r.mean('seconds'):.1f} | {len(r.runs)} | "
                         f"{errors} |")
        lines.append("")
    return lines


def results_markdown(reports: dict[tuple[str, str], KindReport]) -> str:
    lines = ["# Ignite eval results", ""] + results_table(reports, KINDS)
    lines += ["`bm25`: one BM25 search per source with the question text, no model. `retrieval` and "
              "`harness` use the same planner, judge and settings (`config.yaml`); retrieval runs "
              "one planned pass, harness runs the full loop.", "",
              "Small samples: one run per question, 4 single-hop and 7 multi-hop questions. Model "
              "runs vary, so treat differences of a few points as noise.", ""]
    return "\n".join(lines)


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--modes", default="bm25,retrieval,harness")
    args = p.parse_args()

    def harness_for(mode: str) -> Harness:  # a fresh harness per mode: evaluate() closes each
        return bm25_harness() if mode == "bm25" else build_demo_harness(load_config())

    reports = await evaluate(args.modes.split(","), harness_for)
    text = results_markdown(reports)
    RESULTS_PATH.write_text(text)
    print(text)


if __name__ == "__main__":
    asyncio.run(main())
