from agentic_search.core.types import Query
from agentic_search.models.base import PlannerView
from demo.ignite.eval import AllSourcesBM25Driver

SOURCES = ["wiki", "products"]


async def test_bm25_driver_queries_every_source():
    view = PlannerView(question=Query.of("reranker latency"), turn=0, manifest_summary="", digest="")
    plan = await AllSourcesBM25Driver(SOURCES).plan(view, [])
    assert [c.arguments["source"] for c in plan.calls] == SOURCES
    later = await AllSourcesBM25Driver(SOURCES).plan(view.model_copy(update={"turn": 1}), [])
    assert later.calls == []


def _tiny():
    from agentic_search.core.types import Document, TextPart
    from agentic_search.eval.datasets import BeirDataset
    docs = [Document(doc_id=i, content=[TextPart(text=t)])
            for i, t in [("a", "red apple"), ("b", "yellow banana"), ("c", "green pear")]]
    ds = BeirDataset(name="t", corpus={d.doc_id: {"title": "", "text": d.content[0].text}
                                       for d in docs},
                     queries={"q1": "apple"}, qrels={"q1": {"a": 1, "x": 1}})
    return docs, ds


async def test_precision_at_10_counts_gold_in_top_10():
    from demo.ignite.eval import bm25_harness_for, evaluate_groups
    docs, ds = _tiny()
    reports = await evaluate_groups(["bm25"], lambda m: bm25_harness_for({"s": docs}, {"s": "d"}),
                                    ds, {"g": ["q1"]}, name="t")
    assert reports[("bm25", "g")].precision == [0.1]


async def test_results_table_has_p_at_10():
    from demo.ignite.eval import bm25_harness_for, evaluate_groups, results_table
    docs, ds = _tiny()
    reports = await evaluate_groups(["bm25"], lambda m: bm25_harness_for({"s": docs}, {"s": "d"}),
                                    ds, {"g": ["q1"]}, name="t")
    lines = results_table(reports, {"g": "G"})
    header = lines[2].split(" | ")
    row = next(line for line in lines if line.startswith("| bm25 |")).split(" | ")
    assert row[header.index("P@10")] == "0.100"


async def test_evaluate_groups_runs_a_swarm_mode():
    from agentic_search.eval.datasets import BeirDataset
    from demo.ignite.eval import bm25_harness_for, evaluate_groups
    from demo.ignite.tests.test_swarm import make_swarm
    docs, tiny = _tiny()
    ds = BeirDataset(name="t", corpus=tiny.corpus, queries={"q1": "hero question"},
                     qrels={"q1": {"a": 1}})

    def harness_for(mode):
        if mode == "swarm":
            return make_swarm(subs=["a", "b"])
        return bm25_harness_for({"s": docs}, {"s": "d"})

    reports = await evaluate_groups(["bm25", "swarm"], harness_for, ds, {"all": ["q1"]}, name="t")
    assert ("swarm", "all") in reports and reports[("swarm", "all")].turns
    assert reports[("swarm", "all")].report.runs[0].error is None


# ---- wands-multi part scores -------------------------------------------------------------

import pytest  # noqa: E402

P4 = {"a": ["a1", "a2", "a3"], "b": ["b1", "b2"], "c": ["c1"], "d": [f"d{i}" for i in range(20)]}


def test_part_coverage():
    from demo.ignite.eval import part_coverage
    assert part_coverage(["x", "a2", "b1"] + ["x"] * 7 + ["c1"], P4) == 0.5  # c1 is rank 11
    assert part_coverage([], P4) == 0.0


def test_part_recall_caps():
    from demo.ignite.eval import part_recall
    # 4 parts -> per-part cap 7; d has 20 Exact, 7 found -> 1.0; b has 2 Exact, 1 found -> 0.5
    ranked = ["b1"] + [f"d{i}" for i in range(7)]
    assert part_recall(ranked, P4) == pytest.approx((0 + 0.5 + 0 + 1.0) / 4)
    # only the first 30 count
    assert part_recall(["x"] * 30 + ["c1"], {"c": ["c1"]}) == 0.0


def test_shared_product_counts_for_each_part():
    from demo.ignite.eval import part_coverage, part_recall
    assert part_coverage(["s"], {"a": ["s"], "b": ["s"]}) == 1.0
    assert part_recall(["s"], {"a": ["s"], "b": ["s"]}) == 1.0


def _rep(**modes):
    from agentic_search.eval.runner import EvalReport
    from demo.ignite.eval import KindReport
    return {(m, "all"): KindReport(report=EvalReport(name=m, k=10, recall_k=30, runs=[]),
                                   coverage=[c], part_recall=[r])
            for m, (c, r) in modes.items()}


def test_verdict_order():
    from demo.ignite.eval import verdict
    assert verdict(_rep(harness=(0.5, 0.4), swarm=(0.6, 0.4)))[0] == "works"  # exactly 0.10
    assert verdict(_rep(harness=(0.5, 0.4), swarm=(0.62, 0.3),
                        **{"swarm-4x": (0.7, 0.5)}))[0] == "spend"
    assert verdict(_rep(harness=(0.5, 0.4), swarm=(0.62, 0.3)))[0] == "unclear"
    assert verdict(_rep(harness=(0.5, 0.4), swarm=(0.56, 0.5)))[0] == "unclear"
    assert verdict(_rep(harness=(0.5, 0.4), swarm=(0.54, 0.9)))[0] == "no"
    code, sentence = verdict(_rep(harness=(0.5, 0.4), swarm=(0.6, 0.4),
                                  **{"swarm-4x": (0.9, 0.9)}))
    assert code == "works" and "0.600" in sentence and "0.500" in sentence


class _Flaky:
    """bm25 harness that raises for the question "boom"."""

    def __init__(self, inner):
        self.inner = inner

    async def __aenter__(self):
        await self.inner.__aenter__()
        return self

    async def __aexit__(self, *exc):
        return await self.inner.__aexit__(*exc)

    async def search(self, question, **kwargs):
        if str(question) == "boom":
            raise RuntimeError("boom")
        return await self.inner.search(question, **kwargs)


async def test_evaluate_groups_scores_parts_and_errors():
    from agentic_search.eval.datasets import BeirDataset
    from demo.ignite.eval import bm25_harness_for, evaluate_groups
    docs, tiny = _tiny()
    ds = BeirDataset(name="t", corpus=tiny.corpus, queries={"q1": "apple", "q2": "boom"},
                     qrels={"q1": {"a": 2}, "q2": {"b": 2}})
    parts = {"q1": {"p1": ["a"], "p2": ["zz"]}, "q2": {"p1": ["b"]}}
    reports = await evaluate_groups(
        ["harness"], lambda m: _Flaky(bm25_harness_for({"s": docs}, {"s": "d"})), ds,
        {"all": ["q1", "q2"]}, name="t", parts=parts)
    kr = reports[("harness", "all")]
    assert [r.query_id for r in kr.report.runs] == ["q1", "q2"]
    assert kr.coverage == [0.5, 0.0]
    assert kr.part_recall == [0.5, 0.0]


async def test_old_datasets_have_no_part_scores():
    from demo.ignite.eval import bm25_harness_for, evaluate_groups, results_table
    docs, ds = _tiny()
    reports = await evaluate_groups(["bm25"], lambda m: bm25_harness_for({"s": docs}, {"s": "d"}),
                                    ds, {"g": ["q1"]}, name="t")
    assert reports[("bm25", "g")].coverage == [] and reports[("bm25", "g")].part_recall == []
    assert results_table(reports, {"g": "G"})[2] == (
        "| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | "
        "avg cost | avg seconds | n | errors |")


def test_parts_table_columns():
    from demo.ignite.eval import parts_table
    lines = parts_table(_rep(harness=(0.5, 0.4), swarm=(0.75, 0.6)))
    assert lines[0].startswith("| mode | part coverage@10 | part recall@30 | nDCG@10 |")
    assert any(line.startswith("| swarm | 0.750 | 0.600 |") for line in lines)
