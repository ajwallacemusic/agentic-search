import pytest

from agentic_search.config import ConfigError, build_harness, load_harness, register, resolve_env
from agentic_search.models.cross_encoder import CrossEncoderJudge
from agentic_search.models.llm_judge import LLMJudge


def test_resolve_env(monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret")
    assert resolve_env({"a": {"api_key_env": "MY_KEY"}, "b": [1]}) == {"a": {"api_key": "secret"}, "b": [1]}
    monkeypatch.delenv("MY_KEY")
    with pytest.raises(ConfigError, match="MY_KEY"):
        resolve_env({"api_key_env": "MY_KEY"})


def cfg(tmp_path):
    (tmp_path / "notes").mkdir(exist_ok=True)
    (tmp_path / "notes" / "a.md").write_text("aspirin headache")
    return {
        "embedders": [{"type": "hash", "id": "h", "dim": 32}],
        "backends": [{"name": "notes", "type": "files", "root": "notes", "embedder": "h",
                      "allowed_models": ["openai:local"]}],
        "driver": {"type": "openai_compat", "model": "local", "base_url": "http://localhost:8000/v1",
                   "max_calls_per_turn": 4},
        "analyzer": {"type": "cross_encoder", "model": "BAAI/bge-reranker-v2-m3"},
        "controller": {"type": "llm_judge", "client": {"type": "openai_compat", "model": "local",
                                                       "base_url": "http://localhost:8000/v1"}},
        "mode": "retrieval",
        "budget": {"max_turns": 2},
        "settings": {"judge_weight": 0.8},
    }


async def test_build_harness(tmp_path):
    h = build_harness(cfg(tmp_path), base_dir=tmp_path)
    assert h.driver.id == "openai:local" and h.driver.max_calls == 4
    assert isinstance(h.analyzer_decider, CrossEncoderJudge)
    assert isinstance(h.controller_decider, LLMJudge)
    assert h.mode == "retrieval" and h.budget.max_turns == 2 and h.settings.judge_weight == 0.8
    assert h.policy.allows("notes", "openai:local") and not h.policy.allows("notes", "other")
    manifests = await h.setup()
    assert manifests["notes"].resolve_collection(None).field("embedding").embedder_id == "h"


def test_errors(tmp_path):
    bad = cfg(tmp_path)
    bad["backends"][0]["type"] = "nope"
    with pytest.raises(ConfigError, match="unknown backend type 'nope'"):
        build_harness(bad, base_dir=tmp_path)
    no_driver = cfg(tmp_path)
    del no_driver["driver"]
    with pytest.raises(ConfigError, match="driver"):
        build_harness(no_driver, base_dir=tmp_path)
    missing_emb = cfg(tmp_path)
    missing_emb["backends"][0]["embedder"] = "zzz"
    with pytest.raises(ConfigError, match="zzz"):
        build_harness(missing_emb, base_dir=tmp_path)


async def test_load_yaml_with_annotations_file(tmp_path):
    import yaml
    c = cfg(tmp_path)
    c["annotations"] = "ann.yaml"
    (tmp_path / "ann.yaml").write_text(yaml.safe_dump({"notes": {"description": "Clinic notes"}}))
    (tmp_path / "h.yaml").write_text(yaml.safe_dump(c))
    h = load_harness(tmp_path / "h.yaml")
    await h.setup()
    assert h.manifests["notes"].description == "Clinic notes"


def test_register_custom_type(tmp_path):
    made = []
    register("embedder", "custom-test", lambda c, ctx: made.append(c) or __import__(
        "agentic_search.embedders.local", fromlist=["HashEmbedder"]).HashEmbedder(id="custom"))
    c = cfg(tmp_path)
    c["embedders"].append({"type": "custom-test", "x": 1})
    build_harness(c, base_dir=tmp_path)
    assert made == [{"type": "custom-test", "x": 1}]
    with pytest.raises(ValueError):
        register("gizmo", "x", lambda c, ctx: None)


async def test_build_sql_and_search_backends(tmp_path, monkeypatch):
    from agentic_search.backends.bigquery import BigQueryBackend
    from agentic_search.backends.mysql import MySQLBackend
    from agentic_search.backends.opensearch import OpenSearchBackend
    from agentic_search.backends.postgres import PostgresBackend
    from agentic_search.core.secrets import scrub

    monkeypatch.setenv("PG_DSN", "postgresql://app:pg-pass-123@db:5432/app")
    h = build_harness({
        "embedders": [{"type": "hash", "id": "hash64", "dim": 64}],
        "backends": [
            {"name": "pg", "type": "pgvector", "dsn_env": "PG_DSN", "tables": ["docs"],
             "embedders": {"docs.embedding": "hash64"}, "native_query": True},
            {"name": "my", "type": "mysql", "dsn": "mysql://u:my-pass-456@h:3306/app"},
            {"name": "bq", "type": "bigquery", "project": "p", "dataset": "d", "max_bytes_billed": 10},
            {"name": "os", "type": "opensearch", "url": "https://admin:os-pass-789@search:9200",
             "indices": ["docs"], "embedders": {"docs.embedding": "hash64"}},
        ],
        "driver": {"type": "openai_compat", "model": "local", "base_url": "http://localhost:8000/v1"},
    }, base_dir=tmp_path)
    b = h.backends
    assert isinstance(b["pg"], PostgresBackend) and b["pg"].native_query and b["pg"].table_names == ["docs"]
    assert isinstance(b["my"], MySQLBackend) and isinstance(b["os"], OpenSearchBackend)
    assert isinstance(b["bq"], BigQueryBackend) and b["bq"].max_bytes_billed == 10
    leaked = "pg-pass-123 my-pass-456 os-pass-789 https://admin:os-pass-789@search:9200"
    assert all(s not in scrub(leaked) for s in ("pg-pass-123", "my-pass-456", "os-pass-789"))


def test_missing_required_key_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="backend 'postgres' config is missing required key 'dsn'"):
        build_harness({"backends": [{"name": "pg", "type": "postgres"}],
                       "driver": {"type": "openai_compat", "model": "m", "base_url": "http://x/v1"}},
                      base_dir=tmp_path)


def test_timeouts_pass_through(tmp_path):
    h = build_harness({
        "backends": [{"name": "my", "type": "mysql", "dsn": "mysql://u:p@h/db", "statement_timeout_ms": 1234},
                     {"name": "bq", "type": "bigquery", "project": "p", "dataset": "d", "job_timeout_ms": 4321}],
        "driver": {"type": "openai_compat", "model": "m", "base_url": "http://x/v1"}}, base_dir=tmp_path)
    assert h.backends["my"].statement_timeout_ms == 1234 and h.backends["bq"].job_timeout_ms == 4321


def test_backend_embedder_reference_must_exist(tmp_path):
    with pytest.raises(ConfigError, match="unknown embedder"):
        build_harness({"backends": [{"name": "pg", "type": "postgres", "dsn": "postgresql://x@h/db",
                                     "embedders": {"docs.embedding": "nope"}}],
                       "driver": {"type": "openai_compat", "model": "m", "base_url": "http://x/v1"}},
                      base_dir=tmp_path)


def test_constructor_value_errors_become_config_errors(tmp_path):
    with pytest.raises(ConfigError, match="invalid"):
        build_harness({"backends": [{"name": "bq", "type": "bigquery", "project": "bad project!", "dataset": "d"}],
                       "driver": {"type": "openai_compat", "model": "m", "base_url": "http://x/v1"}},
                      base_dir=tmp_path)


async def test_build_remote_embedders_and_typesafe(tmp_path, monkeypatch):
    from agentic_search.embedders.auth import AzureIdentity, Bearer, GcpAdc
    from agentic_search.embedders.http_generic import GenericHttpEmbedder
    from agentic_search.embedders.openai_compat import OpenAICompatEmbedder
    from agentic_search.embedders.tei import TEIEmbedder
    from agentic_search.embedders.vertex import VertexEmbedder
    from agentic_search.models.typesafe import TypeSafeDecider

    monkeypatch.setenv("TS_KEY", "ts-cfg-key-555")
    monkeypatch.setenv("MEDSIGLIP_URL", "https://medsiglip.example/score")
    h = build_harness({
        "embedders": [
            {"type": "openai_compat", "model": "text-embedding-3-small", "dim": 1536,
             "auth": {"type": "bearer", "token": "sk-cfg-1234"}},
            {"type": "tei", "url": "http://tei:8080", "dim": 384, "id": "tei:bge-small"},
            {"type": "vertex", "model": "gemini-embedding-001", "dim": 768, "project": "proj-1"},
            {"type": "http", "id": "azure:medsiglip-448", "url_env": "MEDSIGLIP_URL", "dim": 1152,
             "auth": {"type": "azure_identity", "scope": "api://medsiglip/.default"},
             "request": {"image": {"instances": [{"image_b64": "{{b64}}"}]}},
             "response_path": "$.predictions[*].embedding"},
        ],
        "backends": [{"name": "notes", "type": "files", "root": ".", "glob": "none/*"}],
        "driver": {"type": "openai_compat", "model": "local", "base_url": "http://localhost:8000/v1"},
        "controller": {"type": "typesafe", "api_key_env": "TS_KEY"},
    }, base_dir=tmp_path)
    emb = {e.id: e for e in h.embedders._by_id.values()}
    assert isinstance(emb["openai:text-embedding-3-small"], OpenAICompatEmbedder)
    assert isinstance(emb["openai:text-embedding-3-small"].auth, Bearer)
    assert isinstance(emb["tei:bge-small"], TEIEmbedder)
    assert isinstance(emb["vertex:gemini-embedding-001"].auth, GcpAdc)
    assert isinstance(emb["vertex:gemini-embedding-001"], VertexEmbedder)
    med = emb["azure:medsiglip-448"]
    assert isinstance(med, GenericHttpEmbedder) and isinstance(med.auth, AzureIdentity)
    assert med.url == "https://medsiglip.example/score"
    assert isinstance(h.controller_decider, TypeSafeDecider) and h.controller_decider.id == "typesafe:jev-latest"


def test_build_graph_and_vector_backends(tmp_path, monkeypatch):
    from agentic_search.backends.milvus import MilvusBackend
    from agentic_search.backends.neo4j import Neo4jBackend
    from agentic_search.core.secrets import scrub

    monkeypatch.setenv("NEO4J_PASSWORD", "neo-pass-2468")
    h = build_harness({
        "embedders": [{"type": "hash", "id": "hash64", "dim": 64}],
        "backends": [
            {"name": "kg", "type": "neo4j", "uri": "bolt://graph:7687", "user": "neo4j",
             "password_env": "NEO4J_PASSWORD", "labels": ["Drug", "Condition"],
             "embedders": {"Drug.embedding": "hash64"}, "native_query": True},
            {"name": "vec", "type": "milvus", "uri": "http://milvus:19530", "collections": ["docs"],
             "embedders": {"docs.embedding": "hash64"}},
        ],
        "driver": {"type": "openai_compat", "model": "local", "base_url": "http://localhost:8000/v1"},
    }, base_dir=tmp_path)
    assert isinstance(h.backends["kg"], Neo4jBackend) and isinstance(h.backends["vec"], MilvusBackend)
    assert scrub("neo-pass-2468") == "***"


def test_neo4j_query_timeout_passes_through(tmp_path):
    h = build_harness({
        "embedders": [{"type": "hash", "id": "hash64", "dim": 64}],
        "backends": [{"name": "kg", "type": "neo4j", "uri": "bolt://graph:7687", "query_timeout_s": 4.5}],
        "driver": {"type": "openai_compat", "model": "local", "base_url": "http://localhost:8000/v1"},
    }, base_dir=tmp_path)
    assert h.backends["kg"].query_timeout_s == 4.5
