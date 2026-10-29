from agentic_search.core.types import And, Budget, Eq, FieldType, FilterOnly
from demo.ignite.corpus import SOURCES
from demo.ignite.tests.helpers import offline_harness


def test_one_backend_per_source_in_order():
    assert list(offline_harness().backends) == list(SOURCES)


def test_budget_matches_spec():
    assert offline_harness().budget == Budget(max_turns=4, max_tool_calls=24, max_seconds=60,
                                              max_cost_usd=0.50)


def test_heuristic_controller_is_builtin():
    assert offline_harness().controller_decider is None


async def test_deploys_are_filterable_by_date_and_service():
    manifests = await offline_harness().setup()
    fields = {f.name: f for f in manifests["deploys"].collections[0].fields}
    assert fields["date"].type == FieldType.DATE and fields["date"].filterable
    assert fields["service"].filterable


async def test_filter_search_finds_the_hero_deploy():
    h = offline_harness()
    op = FilterOnly(source="deploys", filter=And(clauses=[Eq(field="service", value="search-api"),
                                        Eq(field="date", value="2026-09-13")]), limit=10)
    hits = await h.backends["deploys"].execute(op)
    assert "DEP-0913-07" in [x.doc_id for x in hits]


def test_sources_are_described():
    assert all(b.description for b in offline_harness().backends.values())
