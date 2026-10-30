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
