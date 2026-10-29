from agentic_search.core.types import Query
from agentic_search.eval.datasets import load_beir
from agentic_search.models.base import PlannerView
from demo.ignite.check_data import HERO_CHAIN
from demo.ignite.corpus import SOURCES, load_questions, load_records
from demo.ignite.eval import AllSourcesBM25Driver, bm25_harness, evaluate, results_markdown
from demo.ignite.export_beir import export_beir


def test_export_round_trips_through_load_beir(tmp_path):
    ds = load_beir(export_beir(tmp_path))
    assert ds.qrels["hero"] == {g: 1 for g in HERO_CHAIN}
    assert set(ds.corpus) == {r["id"] for rows in load_records().values() for r in rows}
    assert set(ds.queries) == set(load_questions())


async def test_bm25_driver_queries_every_source():
    view = PlannerView(question=Query.of("reranker latency"), turn=0, manifest_summary="", digest="")
    plan = await AllSourcesBM25Driver(list(SOURCES)).plan(view, [])
    assert [c.arguments["source"] for c in plan.calls] == list(SOURCES)
    later = await AllSourcesBM25Driver(list(SOURCES)).plan(view.model_copy(update={"turn": 1}), [])
    assert later.calls == []


async def test_evaluate_splits_by_kind(tmp_path):
    reports = await evaluate(["bm25"], lambda mode: bm25_harness(), beir_dir=tmp_path)
    assert set(reports) == {("bm25", "single_hop"), ("bm25", "multi_hop")}
    single = reports[("bm25", "single_hop")]
    assert len(single.report.runs) == sum(q.kind == "single_hop" for q in load_questions().values())
    assert all(t == 1 for t in single.turns)
    table = results_markdown(reports)
    assert "recall@10" in table and "Single-hop" in table and "Multi-hop" in table
    assert "one run per question" in table
