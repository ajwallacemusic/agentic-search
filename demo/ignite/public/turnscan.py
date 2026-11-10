"""Run the harness with a raised turn cap on one public set; save every event per question.
Run: uv run python -m demo.ignite.public.turnscan musique|wands|wands-match
Output: demo/ignite/public/.cache/turnscan/<set>/ (git-ignored). Then: python -m demo.ignite.public.turnsim
"""
import asyncio
import json
import sys
from pathlib import Path

from demo.ignite.build import build_harness_for, load_config
from demo.ignite.public.datasets import sample_musique, sample_wands
from demo.ignite.public.fetch import fetch
from demo.ignite.public.wands_match import sample_wands_match

OUT = Path(__file__).parent / ".cache" / "turnscan"

async def main(name):
    if name == "musique":
        s = sample_musique(fetch("musique") / "musique_ans_v1.0_dev.jsonl")
    elif name == "wands":
        s = sample_wands(fetch("wands"))
    else:
        s = sample_wands_match(fetch("wands"))
    ds = s.to_beir()
    cfg = load_config()
    cfg["budget"] = {"max_turns": 6, "max_tool_calls": 36, "max_seconds": 100, "max_cost_usd": 1.0}
    h = build_harness_for(cfg, {s.source: s.documents}, {s.source: s.description})
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "qrels.json").write_text(json.dumps({q: ds.qrels.get(q, {}) for q in ds.queries}))
    (d / "groups.json").write_text(json.dumps(s.groups))
    sem = asyncio.Semaphore(3)
    async def one(qid):
        f = d / f"{qid.replace('/', '_')}.jsonl"
        if f.exists() and '"search_finished"' in f.read_text():
            return
        async with sem:
            try:
                with f.open("w") as out:
                    async with h.stream(ds.queries[qid], mode="harness", top_k=100, snapshot_k=10) as evs:
                        async for ev in evs:
                            out.write(ev.model_dump_json() + "\n")
            except Exception as e:
                print(qid, "error", e, flush=True)
            print(name, qid, "done", flush=True)
    async with h:
        await asyncio.gather(*(one(q) for q in sorted(ds.queries)))

asyncio.run(main(sys.argv[1]))
