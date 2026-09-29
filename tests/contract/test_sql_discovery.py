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
