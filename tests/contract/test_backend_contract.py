"""Every backend must pass this suite. `files` always runs; the docker-backed params run with
`docker compose up -d --wait` and `AGENTIC_SEARCH_INTEGRATION=1 uv run pytest -m integration`."""

import os

import pytest

from agentic_search import Harness
from agentic_search.backends.base import Backend, BackendError, UnsupportedOperation
from agentic_search.backends.files import FilesBackend
from agentic_search.backends.sql_backend import SqlBackend
from agentic_search.core.types import (
    Aggregate,
    Capability,
    Eq,
    Fetch,
    FilterOnly,
    Hybrid,
    Lexical,
    Native,
    Range,
    Regex,
    StructuredPart,
    TextPart,
    Traverse,
    Vector,
)
from agentic_search.testing import KeywordJudge, ScriptedDriver, call

from . import corpus

INTEGRATION = os.environ.get("AGENTIC_SEARCH_INTEGRATION") == "1"
docker = pytest.mark.skipif(not INTEGRATION, reason="set AGENTIC_SEARCH_INTEGRATION=1 with docker compose up")

NATIVE = {
    "postgres": ("SELECT id FROM docs WHERE type = 'history'", "DELETE FROM docs"),
    "mysql": ("SELECT id FROM docs WHERE type = 'history'", "DELETE FROM docs"),
    "opensearch": ('{"query": {"term": {"type": "history"}}}', '{"query": {"match_all": {}}, "script": {}}'),
}
NATIVE_DIALECT = {"postgres": "sql", "mysql": "sql", "opensearch": "opensearch_dsl"}


async def make_backend(kind: str) -> Backend:
    if kind == "files":
        return FilesBackend.from_documents("files", corpus.documents(), embedder=corpus.EMBEDDER,
                                           collection="docs")
    if kind == "postgres":
        from agentic_search.backends.postgres import PostgresBackend
        await corpus.seed_postgres()
        return PostgresBackend("pg", corpus.PG_DSN, tables=["docs"],
                               embedders={"docs.embedding": "hash64"}, native_query=True)
    if kind == "mysql":
        from agentic_search.backends.mysql import MySQLBackend
        await corpus.seed_mysql()
        return MySQLBackend("my", corpus.MYSQL_DSN, tables=["docs"], native_query=True)
    if kind == "opensearch":
        from agentic_search.backends.opensearch import OpenSearchBackend
        await corpus.seed_opensearch()
        return OpenSearchBackend("os", corpus.OPENSEARCH_URL, indices=["docs"],
                                 embedders={"docs.embedding": "hash64"}, native_query=True)
    raise AssertionError(kind)


@pytest.fixture(params=[
    "files",
    pytest.param("postgres", marks=[pytest.mark.integration, docker]),
    pytest.param("mysql", marks=[pytest.mark.integration, docker]),
    pytest.param("opensearch", marks=[pytest.mark.integration, docker]),
])
async def backend(request):
    b = await make_backend(request.param)
    b.kind = request.param
    yield b
    await b.close()


def ids(hits):
    return [h.doc_id for h in hits]


def text(hit):
    return "\n".join(p.text for p in hit.content if isinstance(p, TextPart))


async def test_discover(backend):
    assert isinstance(backend, Backend)
    m = await backend.discover()
    coll = m.resolve_collection("docs")
    assert coll is not None and coll.count in (None, 5)
    assert {"type", "year"} <= {f.name for f in coll.fields}
    assert {Capability.LEXICAL, Capability.FILTER, Capability.FETCH, Capability.AGGREGATE} <= m.capabilities
    if Capability.VECTOR in m.capabilities:
        emb = coll.field("embedding")
        assert emb.embedder_id == "hash64" and emb.vector_dim == 64


async def test_lexical(backend):
    hits = await backend.execute(Lexical(source=backend.name, collection="docs", text="headache", limit=5))
    assert set(ids(hits)) == {"d1", "d4"}
    assert all(h.raw_score is not None for h in hits)


async def test_lexical_with_filter(backend):
    hits = await backend.execute(Lexical(source=backend.name, collection="docs", text="pain",
                                         filter=Eq(field="year", value=2021)))
    assert ids(hits) == ["d2"]


async def test_filters(backend):
    drugs = await backend.execute(FilterOnly(source=backend.name, collection="docs",
                                             filter=Eq(field="type", value="drug")))
    assert set(ids(drugs)) == {"d1", "d2", "d4"}
    recent = await backend.execute(FilterOnly(source=backend.name, collection="docs",
                                              filter=Range(field="year", gte=2020)))
    assert set(ids(recent)) == {"d1", "d2"}
    two = await backend.execute(FilterOnly(source=backend.name, collection="docs",
                                           filter=Range(field="year", gte=0), limit=2))
    assert len(two) == 2


async def test_vector(backend):
    m = await backend.discover()
    if Capability.VECTOR not in m.capabilities:
        pytest.skip("no vector support")
    [q] = await corpus.EMBEDDER.embed([TextPart(text="headache fever")], "query")
    hits = await backend.execute(Vector(source=backend.name, collection="docs", field="embedding",
                                        hyde_text="x", vector=q, limit=3))
    assert hits[0].doc_id in {"d1", "d4"}


async def test_hybrid(backend):
    m = await backend.discover()
    if Capability.HYBRID not in m.capabilities:
        pytest.skip("no hybrid support")
    [q] = await corpus.EMBEDDER.embed([TextPart(text="headache fever")], "query")
    hits = await backend.execute(Hybrid(source=backend.name, collection="docs", text="headache",
                                        field="embedding", vector=q, limit=3))
    assert hits and hits[0].doc_id in {"d1", "d4"}
    assert all(h.raw_score is not None for h in hits)


async def test_regex_fetch_aggregate(backend):
    rx = await backend.execute(Regex(source=backend.name, collection="docs", pattern="print.*"))
    assert ids(rx) == ["d3"]
    [f] = await backend.execute(Fetch(source=backend.name, collection="docs", doc_ids=["d3"]))
    assert f.doc_id == "d3" and "printing" in text(f).lower()
    agg = await backend.execute(Aggregate(source=backend.name, collection="docs", group_by=["type"]))
    counts = {h.content[0].data["type"]: h.content[0].data["count"] for h in agg
              if isinstance(h.content[0], StructuredPart)}
    assert counts == {"drug": 3, "history": 2}


async def test_native_read_only(backend):
    m = await backend.discover()
    if Capability.NATIVE not in m.capabilities:
        pytest.skip("native queries not enabled")
    select, write = NATIVE[backend.kind]
    dialect = NATIVE_DIALECT[backend.kind]
    rows = await backend.execute(Native(source=backend.name, collection="docs", dialect=dialect, query=select))
    assert len(rows) == 2
    with pytest.raises(BackendError):
        await backend.execute(Native(source=backend.name, collection="docs", dialect=dialect, query=write))


async def test_sql_sessions_are_read_only(backend):
    if not isinstance(backend, SqlBackend):
        pytest.skip("not a SQL backend")
    await backend.discover()
    with pytest.raises(BackendError):
        await backend._query("DELETE FROM docs WHERE id = 'd1'", None)
    [still] = await backend.execute(Fetch(source=backend.name, collection="docs", doc_ids=["d1"]))
    assert still.doc_id == "d1"


async def test_unsupported_op(backend):
    with pytest.raises((UnsupportedOperation, BackendError)):
        await backend.execute(Traverse(source=backend.name, collection="docs", start=Eq(field="type", value="x")))


async def test_harness_end_to_end(backend):
    driver = ScriptedDriver([[call("lexical_search", source=backend.name, collection="docs", text="headache")]])
    h = Harness([backend], driver, embedders=[corpus.EMBEDDER], analyzer=KeywordJudge(["headache"]))
    res = await h.search("what treats headache?")
    assert set(res.keys()) == {f"{backend.name}:d1", f"{backend.name}:d4"}
    assert all(r.judged and r.p_relevant == 1.0 for r in res.hits)
