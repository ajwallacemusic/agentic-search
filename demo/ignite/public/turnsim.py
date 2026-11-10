"""Offline stop-rule test on saved 6-turn harness runs (turnscan.py output).

For each question: the top 10 after each turn's judge, plus what the controller could see then.
A stop rule picks a turn; quality there is exact, because later turns never change earlier ones.
Each table is printed for all questions and for two fixed halves (hash of the query id), so a
rule that only wins on one half shows up.

Run: uv run python -m demo.ignite.public.turnsim [folder]   (see tuning-log.md, round 14)
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).parent / ".cache" / "turnscan"
REL = 0.5  # judge threshold, as HarnessSettings.relevant_threshold
SETS = ("musique", "wands", "wands-match")

Turn = dict[str, Any]
Rule = Callable[[list[Turn]], bool]


def ndcg(ids: list[str], qrels: dict[str, int], k: int = 10) -> float:
    dcg = sum((2 ** qrels.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ids[:k]))
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def load(folder: Path) -> dict[str, list[Turn]]:
    """Query id -> one record per turn: top-10 ids, judged-relevant count, step state, scores."""
    qrels = json.loads((folder / "qrels.json").read_text())
    out: dict[str, list[Turn]] = {}
    for f in sorted(folder.glob("*.jsonl")):
        events = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        if not any(e["type"] == "search_finished" for e in events):
            continue
        by_turn: dict[int, Turn] = {}
        steps_open = False  # no plan counts as no open step
        for e in events:
            if e["type"] == "steps_updated":
                steps_open = e.get("current") is not None
            elif e["type"] == "results_updated":
                hits = e["hits"][:10]
                by_turn[e["turn"]] = {
                    "turn": e["turn"], "ids": [h["doc_id"] for h in hits],
                    "rel": sum(1 for h in hits if (h.get("p_relevant") or 0) >= REL),
                    "steps_open": steps_open, "sec": e["at_ms"] / 1000}
        if not by_turn:
            continue
        qr = qrels.get(f.stem) or qrels.get(f.stem.replace("_", "/"), {})
        n_gold = max(1, sum(1 for v in qr.values() if v > 0))
        turns = [by_turn[k] for k in sorted(by_turn)]
        for t in turns:
            right = sum(1 for d in t["ids"] if qr.get(d, 0) > 0)
            t["p10"], t["r10"], t["ndcg"] = right / 10, right / n_gold, ndcg(t["ids"], qr)
        out[f.stem] = turns
    return out


def gain(ts: list[Turn]) -> int:
    return ts[-1]["rel"] - (ts[-2]["rel"] if len(ts) > 1 else 0)


def flat(ts: list[Turn]) -> bool:
    return len(ts) > 1 and gain(ts) <= 0


RULES: dict[str, Rule] = {
    "never (run to cap)": lambda ts: False,
    "top 10 unchanged": lambda ts: len(ts) > 1 and set(ts[-1]["ids"]) == set(ts[-2]["ids"]),
    "no step open + flat": lambda ts: not ts[-1]["steps_open"] and flat(ts),
    "no step open + >=8 rel": lambda ts: not ts[-1]["steps_open"] and ts[-1]["rel"] >= 8,
    "CHOSEN: no step open + (>=8 rel or flat)":
        lambda ts: not ts[-1]["steps_open"] and (ts[-1]["rel"] >= 8 or flat(ts)),
    "flat (ignore steps)": flat,
    ">=8 rel (ignore steps)": lambda ts: ts[-1]["rel"] >= 8,
}


def pick(turns: list[Turn], rule: Rule, cap: int) -> Turn:
    last = min(len(turns), cap)
    for i in range(last):
        if rule(turns[: i + 1]):
            return turns[i]
    return turns[last - 1]


def table(data: dict[str, list[Turn]], label: str, cap: int) -> None:
    print(f"\n### {label}, cap {cap} (n={len(data)})\n")
    print("| rule | P@10 | recall@10 | nDCG@10 | avg turns | avg s |")
    print("|---|---|---|---|---|---|")
    for name, rule in RULES.items():
        picks = [pick(turns, rule, cap) for turns in data.values()]
        n = len(picks) or 1
        mean = lambda key: sum(p[key] for p in picks) / n  # noqa: E731
        print(f"| {name} | {mean('p10'):.3f} | {mean('r10'):.3f} | {mean('ndcg'):.3f} | "
              f"{sum(p['turn'] + 1 for p in picks) / n:.2f} | {mean('sec'):.1f} |")


def half(qid: str, h: int) -> bool:
    return int(hashlib.md5(qid.encode()).hexdigest(), 16) % 2 == h


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT
    for name in SETS:
        if not (root / name).exists():
            continue
        data = load(root / name)
        for cap in (6, 4):
            table(data, f"{name}, all", cap)
            for h in (0, 1):
                table({q: t for q, t in data.items() if half(q, h)}, f"{name}, half {h}", cap)


if __name__ == "__main__":
    main()
