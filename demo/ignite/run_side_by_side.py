"""Run one demo question in retrieval mode, then harness mode, and compare.

Run: uv run python -m demo.ignite.run_side_by_side --question hero [--controller typesafe] [--save]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from agentic_search.core.harness import Harness, Mode
from agentic_search.events import (
    PhaseFinished,
    PhaseStarted,
    SearchEvent,
    SearchFailed,
    SearchFinished,
    ToolCallFinished,
    ToolCallStarted,
    UsageUpdated,
)
from demo.ignite.build import build_demo_harness, load_config
from demo.ignite.corpus import Question, corpus_of, load_demo

RUNS_DIR = Path(__file__).parent / "runs"
GOLD_K = 10
ARGS_CHARS = 80


class RunSummary(BaseModel):
    mode: str
    turns: int
    tool_calls: int
    seconds: float
    cost_usd: float
    stop_reason: str
    top5: list[str]
    gold_recall: int
    gold_total: int


def gold_recall(doc_ids: list[str], gold: list[str], k: int = GOLD_K) -> int:
    return len(set(gold) & set(doc_ids[:k]))


def _short_args(arguments: dict[str, Any]) -> str:
    shown = {k: v for k, v in arguments.items() if k != "source"}
    text = json.dumps(shown, separators=(", ", ": "), default=str)
    return text if len(text) <= ARGS_CHARS else text[: ARGS_CHARS - 1] + "…"


def format_event(ev: SearchEvent) -> str | None:
    """One compact trace line, or None for events the trace does not show."""
    if isinstance(ev, PhaseStarted):
        return f"  [turn {ev.turn}] … {ev.phase}"
    if isinstance(ev, ToolCallStarted):
        return f"      → {ev.source or '-'}.{ev.name} {_short_args(ev.arguments)}"
    if isinstance(ev, ToolCallFinished) and ev.error is not None:
        return f"      ✗ {ev.error.kind}: {ev.error.message[:ARGS_CHARS]}"
    if isinstance(ev, PhaseFinished) and ev.phase == "decide" and ev.summary.action:
        return f"  ◆ {ev.summary.action}: {ev.summary.note or ''}"
    if isinstance(ev, SearchFailed):
        return f"  ✗ search failed: {ev.error_type}: {ev.message}"
    return None


async def run_one(harness: Harness, question: Question, mode: Mode, *, save_to: Path | None,
                  print_fn: Callable[[str], None] = print) -> RunSummary:
    """Stream one search, printing the trace and (if `save_to`) writing each event as it comes."""
    out = save_to.open("w", encoding="utf-8") if save_to is not None else None
    t0 = time.perf_counter()
    usage, finished, failed = None, None, None
    try:
        async with harness.stream(question.text, mode=mode, top_k=20, snapshot_k=10) as events:
            async for ev in events:
                if out is not None:
                    out.write(ev.model_dump_json() + "\n")
                    out.flush()
                line = format_event(ev)
                if line is not None:
                    print_fn(line)
                if isinstance(ev, UsageUpdated):
                    usage = ev.usage
                elif isinstance(ev, SearchFinished):
                    finished = ev
                elif isinstance(ev, SearchFailed):
                    failed = ev
    finally:
        if out is not None:
            out.close()
    seconds = time.perf_counter() - t0
    if finished is not None:
        usage = finished.result.usage
        doc_ids = [h.hit.doc_id for h in finished.result.hits]
        top5 = [h.hit.key for h in finished.result.hits[:5]]
        stop = finished.stop_reason.value
    else:
        doc_ids, top5 = [], []
        stop = f"failed: {failed.error_type}" if failed is not None else "failed: no result"
    return RunSummary(mode=mode, turns=usage.turns if usage else 0,
                      tool_calls=usage.tool_calls if usage else 0, seconds=seconds,
                      cost_usd=usage.cost_usd if usage else 0.0, stop_reason=stop, top5=top5,
                      gold_recall=gold_recall(doc_ids, question.gold), gold_total=len(question.gold))


def save_path(question_id: str, mode: str, stamp: str) -> Path:
    return RUNS_DIR / f"{stamp}-{question_id}-{mode}.jsonl"


def write_meta(question: Question, stamp: str) -> Path:
    path = RUNS_DIR / f"{stamp}-{question.id}.meta.json"
    path.write_text(json.dumps({"question_id": question.id, "set": question.set,
                                "ref": question.ref, "text": question.text,
                                "gold": question.gold}, indent=1))
    return path


def summary_table(rows: list[RunSummary]) -> str:
    head = (f"{'mode':<10} {'turns':>5} {'calls':>5} {'secs':>6} {'cost':>8} "
            f"{'gold@10':>7}  stop")
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r.mode:<10} {r.turns:>5} {r.tool_calls:>5} {r.seconds:>6.1f} "
                     f"${r.cost_usd:>7.4f} {r.gold_recall:>3}/{r.gold_total:<3}  {r.stop_reason}")
    for r in rows:
        lines.append(f"\n{r.mode} top 5:")
        lines += [f"  {i + 1}. {key}" for i, key in enumerate(r.top5)] or ["  (none)"]
    return "\n".join(lines)


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--question", default="hero")
    p.add_argument("--controller", choices=["heuristic", "typesafe"], default="heuristic")
    p.add_argument("--save", action="store_true", help="write every event to runs/")
    args = p.parse_args()
    corpus = corpus_of(args.question)
    data = load_demo(corpora={corpus})[corpus]
    question = data.questions[args.question]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.save:
        RUNS_DIR.mkdir(exist_ok=True)
        write_meta(question, stamp)
    rows = []
    async with build_demo_harness(load_config(), data=data, controller=args.controller) as harness:
        await harness.setup()
        for mode in ("retrieval", "harness"):
            print(f"\n=== {mode} mode: {question.text}")
            save_to = save_path(question.id, mode, stamp) if args.save else None
            rows.append(await run_one(harness, question, mode, save_to=save_to))
    print("\n" + summary_table(rows))


if __name__ == "__main__":
    asyncio.run(main())
