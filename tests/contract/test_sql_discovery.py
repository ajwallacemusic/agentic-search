"""Docker-backed discovery edge cases: odd column types and multi-collection sources. Extra tables
and indices are created with separate write connections and dropped afterwards."""

import pytest

from . import corpus
from .test_backend_contract import docker

pytestmark = [pytest.mark.integration, docker]


async def _pg_exec(*statements: str) -> None:
    import psycopg

    async with await psycopg.AsyncConnection.connect(corpus.PG_DSN, autocommit=True) as conn:
        for s in statements:
            await conn.execute(s)


async def test_postgres_discovery_survives_unsampleable_columns():
    from agentic_search.backends.postgres import PostgresBackend

    await _pg_exec("DROP TABLE IF EXISTS odd_types",
                   "CREATE TABLE odd_types (id TEXT PRIMARY KEY, loc point, doc xml, kind VARCHAR(10))",
                   "INSERT INTO odd_types VALUES ('a', point(1,2), '<x/>', 'k1'), ('b', point(3,4), '<y/>', 'k2')")
    b = PostgresBackend("pg", corpus.PG_DSN, tables=["odd_types"])
    try:
        m = await b.discover()
        coll = m.resolve_collection("odd_types")
        assert coll is not None
        assert coll.field("loc").sample_values is None and coll.field("doc").sample_values is None
        assert sorted(coll.field("kind").sample_values) == ["k1", "k2"]
    finally:
        await b.close()
        await _pg_exec("DROP TABLE IF EXISTS odd_types")


async def _harness_keys(backend, calls) -> set[str]:
    from agentic_search import Harness
    from agentic_search.testing import KeywordJudge, ScriptedDriver

    h = Harness([backend], ScriptedDriver([calls]), embedders=[corpus.EMBEDDER],
                analyzer=KeywordJudge(["headache"]))
    return set((await h.search("what treats headache?")).keys())


async def test_postgres_multi_table_ids_are_namespaced():
    from agentic_search.backends.postgres import PostgresBackend
    from agentic_search.core.types import Fetch, FilterOnly, Lexical
    from agentic_search.testing import call

    await corpus.seed_postgres()
    await _pg_exec("DROP TABLE IF EXISTS authors", "DROP TABLE IF EXISTS pairs",
                   "CREATE TABLE authors (id TEXT PRIMARY KEY, name TEXT, bio TEXT)",
                   "INSERT INTO authors VALUES ('d1', 'Ann', 'writes about headache remedies'), "
                   "('d3', 'Bob', 'writes about castles')",
                   "CREATE TABLE pairs (a TEXT, b TEXT, note TEXT, PRIMARY KEY (a, b))")
    b = PostgresBackend("pg", corpus.PG_DSN, tables=["docs", "authors", "pairs"])
    b2 = PostgresBackend("pg2", corpus.PG_DSN, tables=["pairs"], id_columns={"pairs": "a"})
    try:
        m = await b.discover()
        assert {c.name for c in m.collections} == {"docs", "authors"}
        assert "pairs (composite primary key" in m.description
        docs = await b.execute(Lexical(source="pg", collection="docs", text="headache"))
        authors = await b.execute(FilterOnly(source="pg", collection="authors",
                                             filter={"op": "exists", "field": "bio"}))
        assert {h.doc_id for h in docs} == {"docs/d1", "docs/d4"}
        assert {h.doc_id for h in authors} == {"authors/d1", "authors/d3"}
        [f] = await b.execute(Fetch(source="pg", doc_ids=["docs/d3"]))
        assert f.doc_id == "docs/d3"
        [f] = await b.execute(Fetch(source="pg", collection="authors", doc_ids=["d3"]))
        assert f.doc_id == "authors/d3"
        keys = await _harness_keys(b, [call("lexical_search", source="pg", collection="docs", text="headache"),
                                       call("lexical_search", source="pg", collection="authors",
                                            text="headache")])
        assert keys == {"pg:docs/d1", "pg:docs/d4", "pg:authors/d1"}
        m2 = await b2.discover()
        assert [c.name for c in m2.collections] == ["pairs"]
    finally:
        await b.close()
        await b2.close()
        await _pg_exec("DROP TABLE IF EXISTS authors", "DROP TABLE IF EXISTS pairs")


async def test_opensearch_multi_index_ids_are_namespaced():
    from opensearchpy import AsyncOpenSearch

    from agentic_search.backends.opensearch import OpenSearchBackend
    from agentic_search.core.types import Fetch, Lexical
    from agentic_search.testing import call

    await corpus.seed_opensearch()
    admin = AsyncOpenSearch(hosts=[corpus.OPENSEARCH_URL])
    b = OpenSearchBackend("os", corpus.OPENSEARCH_URL, indices=["docs", "authors"],
                          embedders={"docs.embedding": "hash64"})
    try:
        if await admin.indices.exists(index="authors"):
            await admin.indices.delete(index="authors")
        await admin.indices.create(index="authors", body={
            "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
            "mappings": {"properties": {"name": {"type": "text"}, "bio": {"type": "text"}}}})
        await admin.index(index="authors", id="d1", body={"name": "Ann", "bio": "headache remedies"},
                          refresh=True)
        docs = await b.execute(Lexical(source="os", collection="docs", text="headache"))
        authors = await b.execute(Lexical(source="os", collection="authors", text="headache"))
        assert {h.doc_id for h in docs} == {"docs/d1", "docs/d4"}
        assert [h.doc_id for h in authors] == ["authors/d1"]
        [f] = await b.execute(Fetch(source="os", doc_ids=["docs/d3"]))
        assert f.doc_id == "docs/d3"
        [f] = await b.execute(Fetch(source="os", collection="authors", doc_ids=["d1"]))
        assert f.doc_id == "authors/d1"
        keys = await _harness_keys(b, [call("lexical_search", source="os", collection="docs", text="headache"),
                                       call("lexical_search", source="os", collection="authors",
                                            text="headache")])
        assert keys == {"os:docs/d1", "os:docs/d4", "os:authors/d1"}
    finally:
        await b.close()
        await admin.indices.delete(index="authors", ignore_unavailable=True)
        await admin.close()


async def test_postgres_uses_stored_tsvector_column():
    from agentic_search.backends.postgres import PostgresBackend
    from agentic_search.core.types import Lexical

    await _pg_exec(
        "DROP TABLE IF EXISTS tsv_docs",
        "CREATE TABLE tsv_docs (id TEXT PRIMARY KEY, title TEXT, body TEXT, search tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', coalesce(title, '') || ' ' || coalesce(body, ''))) STORED)",
        "CREATE INDEX tsv_docs_search ON tsv_docs USING GIN (search)",
        *[f"INSERT INTO tsv_docs (id, title, body) VALUES ('{i}', '{t}', '{b}')" for i, t, b, _, _ in corpus.ROWS])
    b = PostgresBackend("pg", corpus.PG_DSN, tables=["tsv_docs"])
    try:
        m = await b.discover()
        assert b._tsv == {"tsv_docs": "search"}
        assert m.resolve_collection("tsv_docs").field("search") is None  # not exposed as a field
        hits = await b.execute(Lexical(source="pg", collection="tsv_docs", text="headache"))
        assert {h.doc_id for h in hits} == {"d1", "d4"} and all(h.raw_score for h in hits)
    finally:
        await b.close()
        await _pg_exec("DROP TABLE IF EXISTS tsv_docs")
