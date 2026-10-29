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


async def test_serial_embedder_never_runs_two_calls_at_once():
    import asyncio

    from agentic_search.core.types import Modality, TextPart
    from demo.ignite.build import SerialEmbedder

    class Tracking:
        id, dim, local, modalities = "t", 3, True, {Modality.TEXT}
        active = peak = 0

        async def embed(self, items, purpose):
            Tracking.active += 1
            Tracking.peak = max(Tracking.peak, Tracking.active)
            await asyncio.sleep(0.01)
            Tracking.active -= 1
            return [[1.0, 0.0, 0.0] for _ in items]

    e = SerialEmbedder(Tracking())
    await asyncio.gather(*(e.embed([TextPart(text="x")], "document") for _ in range(5)))
    assert Tracking.peak == 1
    assert e.id == "t" and e.local is True and e.dim == 3


def test_harness_settings_come_from_config():
    from demo.ignite.build import load_config

    cfg = load_config()
    s = offline_harness().settings
    for key, value in cfg["settings"].items():
        assert getattr(s, key) == value


def test_each_collection_is_named_after_its_source():
    assert all(b.collection == name for name, b in offline_harness().backends.items())


async def test_request_defaults_are_added_to_every_completion_call():
    from demo.ignite.build import RequestDefaults

    seen = {}

    class Inner:
        class chat:
            class completions:
                @staticmethod
                async def create(**kwargs):
                    seen.update(kwargs)
                    return "ok"

    client = RequestDefaults(Inner(), reasoning_effort="low")
    assert await client.chat.completions.create(model="m") == "ok"
    assert seen == {"model": "m", "reasoning_effort": "low"}


def test_vertex_driver_gets_request_defaults(monkeypatch):
    from demo.ignite.build import RequestDefaults, make_client

    monkeypatch.setenv("IGNITE_GCP_PROJECT", "demo-project-1")
    client = make_client({"type": "vertex", "model": "gemini-3.8-flash",
                          "project_env": "IGNITE_GCP_PROJECT", "price_per_mtok": [0.75, 3.75],
                          "reasoning_effort": "low", "request_timeout_s": 30, "max_retries": 1})
    assert isinstance(client._client, RequestDefaults)
    assert client._client.defaults == {"reasoning_effort": "low"}
    assert client.price == (0.75, 3.75) and client.id == "vertex:gemini-3.8-flash"


def test_judge_can_use_its_own_model(monkeypatch):
    from demo.ignite.build import build_demo_harness, load_config

    monkeypatch.setenv("IGNITE_GCP_PROJECT", "demo-project-1")
    cfg = load_config()
    cfg["driver"] = {**cfg["driver"], "model": "gemini-3.8-flash"}
    cfg["judge"] = {**cfg["driver"], "model": "gemini-3.5-flash-lite"}
    from agentic_search.embedders.local import HashEmbedder
    h = build_demo_harness(cfg, embedder=HashEmbedder())
    assert h.driver.id == "vertex:gemini-3.8-flash"
    assert h.analyzer_decider.id.endswith("gemini-3.5-flash-lite")


def test_a_stalled_turn_is_cut_short():
    from demo.ignite.build import load_config

    cfg = load_config()
    s = offline_harness().settings
    assert s.call_timeout <= 15 and s.decider_timeout <= 15
    for role in ("driver", "judge"):
        assert cfg[role]["request_timeout_s"] * (cfg[role]["max_retries"] + 1) <= 30


def test_source_descriptions_give_the_year():
    from demo.ignite.build import SOURCE_DESCRIPTIONS

    assert all("2026" in d for d in SOURCE_DESCRIPTIONS.values())
