from demo.ignite.public.datasets import sample_musique
from demo.ignite.public.eval_public import write_results
from demo.ignite.tests.test_public_datasets import _musique


async def test_eval_public_bm25_on_fixture_writes_tables(tmp_path):
    path, _ = _musique(tmp_path)
    sample = sample_musique(path, per_hop=2, corpus_size=40)
    out = tmp_path / "results-musique.md"
    text = await write_results(sample, ["bm25"], out, settings={"per_hop": 2, "seed": 7},
                               hashes={"musique_ans_v1.0_dev.jsonl": "abc"})
    assert out.read_text() == text
    for group in ("2hop", "3hop", "4hop", "All"):
        assert f"## {group}" in text
    assert "| bm25 |" in text and "nDCG@10" in text and "abc" in text and "per_hop" in text
    assert "concurrency 1" in text and "not directly comparable" in text


async def test_all_row_pools_the_group_runs(tmp_path):
    from demo.ignite.eval import evaluate_groups
    from demo.ignite.public.eval_public import _harness_for, _with_all

    path, _ = _musique(tmp_path)
    sample = sample_musique(path, per_hop=2, corpus_size=40)
    reports = await evaluate_groups(["bm25"], _harness_for(sample), sample.to_beir(),
                                    sample.groups, name="t")
    pooled = _with_all(reports, sample.groups)[("bm25", "all")]
    runs = [r for g in sample.groups for r in reports[("bm25", g)].report.runs]
    assert len(pooled.report.runs) == 6
    assert pooled.report.mean("recall") == sum(r.recall for r in runs) / len(runs)


async def test_all_row_pools_precision(tmp_path):
    from demo.ignite.eval import evaluate_groups
    from demo.ignite.public.eval_public import _harness_for, _with_all

    path, _ = _musique(tmp_path)
    sample = sample_musique(path, per_hop=2, corpus_size=40)
    reports = await evaluate_groups(["bm25"], _harness_for(sample), sample.to_beir(),
                                    sample.groups, name="t")
    pooled = _with_all(reports, sample.groups)[("bm25", "all")]
    assert pooled.precision == [p for g in sample.groups for p in reports[("bm25", g)].precision]
    assert len(pooled.precision) == 6


def test_eval_public_accepts_swarm_mode():
    from demo.ignite.public.eval_public import build_parser
    args = build_parser().parse_args(["--dataset", "musique", "--modes", "harness,swarm"])
    assert args.modes == "harness,swarm"



def test_scaled_budget_cfg_does_not_mutate():
    from demo.ignite.public.eval_public import scaled_budget_cfg
    cfg = {"budget": {"max_turns": 4, "max_tool_calls": 24, "max_seconds": 120,
                      "max_cost_usd": 0.50}, "swarm": {"agent_max_turns": 2}}
    out = scaled_budget_cfg(cfg, 4, 4)
    assert out["budget"] == {"max_turns": 4, "max_tool_calls": 96, "max_seconds": 120,
                             "max_cost_usd": 2.00}
    assert out["swarm"] == {"agent_max_turns": 4}
    assert cfg["budget"]["max_tool_calls"] == 24 and cfg["swarm"]["agent_max_turns"] == 2


def test_parser_accepts_wands_multi_and_out():
    from demo.ignite.public.eval_public import build_parser
    args = build_parser().parse_args(["--dataset", "wands-multi", "--n", "2", "--out", "x.md"])
    assert args.dataset == "wands-multi" and args.n == 2 and args.out == "x.md"


async def test_write_results_wands_multi_has_parts_and_verdict(tmp_path):
    from agentic_search.core.types import Document, TextPart
    from demo.ignite.eval import bm25_harness_for
    from demo.ignite.public.datasets import PublicSample

    docs = [Document(doc_id=i, content=[TextPart(text=t)])
            for i, t in [("a", "teak chair"), ("b", "patio umbrella"), ("c", "outdoor rug")]]
    sample = PublicSample(name="wands-multi", source="s", description="d", documents=docs,
                          queries={"m01": "teak chair; patio umbrella"},
                          qrels={"m01": {"a": 2, "b": 2}}, groups={"all": ["m01"]},
                          parts={"m01": {"1": ["a"], "2": ["b"]}})
    text = await write_results(sample, ["harness", "swarm", "swarm-4x"], tmp_path / "r.md",
                               settings={}, hashes={},
                               harness_for=lambda m: bm25_harness_for({"s": docs}, {"s": "d"}))
    assert "part coverage@10" in text and "## Verdict" in text and "swarm-4x" in text
    assert "| swarm-4x |" in text
