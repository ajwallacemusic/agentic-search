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
