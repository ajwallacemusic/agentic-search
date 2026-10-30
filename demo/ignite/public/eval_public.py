"""Score bm25, retrieval and harness modes on a public dataset sample; write results-<name>.md.

Run: uv run python -m demo.ignite.public.eval_public --dataset musique|wands|wands-match
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any, Callable

from agentic_search.core.harness import Harness
from agentic_search.eval.runner import EvalReport
from demo.ignite.build import build_harness_for, load_config
from demo.ignite.eval import KindReport, bm25_harness_for, evaluate_groups, results_table
from demo.ignite.public.datasets import PublicSample, sample_musique, sample_wands
from demo.ignite.public.fetch import FILES, fetch, sha256_of
from demo.ignite.public.wands_match import LABELS_PATH, MATCH_PATH, sample_wands_match

RESULTS_DIR = Path(__file__).parent.parent

ABOUT = {
    "musique": ("MuSiQue v1.0 (Trivedi et al., TACL 2022, CC BY 4.0): multi-hop questions over "
                "Wikipedia paragraphs, built so a single-hop shortcut cannot answer them. Gold: "
                "the supporting paragraphs. recall@10 = share of supporting paragraphs in the top "
                "10. Distractors come from other questions and are unlabeled, so a near-duplicate "
                "of a gold paragraph counts as a miss for every mode alike."),
    "wands": ("WANDS (Wayfair, Chen et al., ECIR 2022, MIT): product search with human labels. "
              "Gold: Exact = 2, Partial = 1. Many queries have dozens of Exact products, so "
              "recall@10 stays low for every mode; nDCG@10 is the headline metric."),
    "wands-match": ("WANDS match: our own questions over the WANDS catalogue (Wayfair, MIT). Each "
                    "names a product the shopper owns and asks for other product types that match "
                    "it on wood tone, color or style. Questions and rules are ours (picked by a "
                    "seeded rule before any model ran, `wands_match.yaml`); labels are built from "
                    "the rules, not by a model: gold = a target product type with the anchor's "
                    "value. Most questions have more than 10 right answers, so P@10 (share of "
                    "the top 10 that is right) is the headline metric."),
}


def _harness_for(sample: PublicSample) -> Callable[[str], Harness]:
    docs = {sample.source: sample.documents}
    descriptions = {sample.source: sample.description}

    def harness_for(mode: str) -> Harness:
        if mode == "bm25":
            return bm25_harness_for(docs, descriptions)
        return build_harness_for(load_config(), docs, descriptions)
    return harness_for


def _with_all(reports: dict[tuple[str, str], KindReport],
              groups: dict[str, list[str]]) -> dict[tuple[str, str], KindReport]:
    """Add an "all" row per mode by pooling the group runs (no queries are run twice)."""
    if len(groups) < 2:
        return reports
    out = dict(reports)
    for mode in dict.fromkeys(m for m, _ in reports):
        parts = [reports[(mode, g)] for g in groups]
        first = parts[0].report
        runs = [r for p in parts for r in p.report.runs]
        out[(mode, "all")] = KindReport(
            report=EvalReport(name=f"{first.name.rsplit('/', 1)[0]}/all", k=first.k,
                              recall_k=first.recall_k, runs=runs),
            turns=[t for p in parts for t in p.turns],
            precision=[x for p in parts for x in p.precision])
    return out


async def write_results(sample: PublicSample, modes: list[str], out: Path, *,
                        settings: dict[str, Any], hashes: dict[str, str],
                        harness_for: Callable[[str], Harness] | None = None) -> str:
    reports = await evaluate_groups(modes, harness_for or _harness_for(sample), sample.to_beir(),
                                    sample.groups, name=sample.name)
    reports = _with_all(reports, sample.groups)
    titles = {g: g for g in sample.groups}
    if len(sample.groups) > 1 or "all" in sample.groups:
        titles["all"] = "All"
    lines = [f"# Public eval: {sample.name}", "", ABOUT.get(sample.name, ""), "",
             f"Sample: {len(sample.queries)} queries, {len(sample.documents)} documents in one "
             f"source (`{sample.source}`). Settings: "
             + ", ".join(f"`{k}={v}`" for k, v in settings.items()) + ".", "",
             "Source files (SHA-256): " + ", ".join(f"`{n}` {h}" for n, h in hashes.items()), ""]
    lines += results_table(reports, titles)
    lines += ["`bm25`: one BM25 search with the question text, no model. `retrieval` and `harness` "
              "use the same planner, judge and settings as the Ignite demo (`config.yaml`).", "",
              "One run per query. Model runs vary, so treat differences of a few points as noise.",
              ""]
    text = "\n".join(lines)
    Path(out).write_text(text)
    return text


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", choices=[*sorted(FILES), "wands-match"], required=True)
    p.add_argument("--n", type=int, help="musique: questions per hop count (20); wands: queries (60)")
    p.add_argument("--corpus-size", type=int, default=5000, help="musique only")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--modes", default="bm25,retrieval,harness")
    args = p.parse_args()
    source = "wands" if args.dataset == "wands-match" else args.dataset
    folder = fetch(source)
    hashes = {name: sha256_of(folder / name) for name in FILES[source]}
    if args.dataset == "wands-match":
        settings = {"questions": MATCH_PATH.name, "labels": LABELS_PATH.name}
        hashes = {"product.csv": hashes["product.csv"], MATCH_PATH.name: sha256_of(MATCH_PATH),
                  LABELS_PATH.name: sha256_of(LABELS_PATH)}
        sample = sample_wands_match(folder)
    elif args.dataset == "musique":
        settings = {"per_hop": args.n or 20, "corpus_size": args.corpus_size, "seed": args.seed}
        sample = sample_musique(folder / "musique_ans_v1.0_dev.jsonl", **settings)
    else:
        settings = {"n": args.n or 60, "seed": args.seed}
        sample = sample_wands(folder, **settings)
    out = RESULTS_DIR / f"results-{args.dataset}.md"
    print(await write_results(sample, args.modes.split(","), out, settings=settings,
                              hashes=hashes))


if __name__ == "__main__":
    asyncio.run(main())
