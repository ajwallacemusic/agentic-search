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
