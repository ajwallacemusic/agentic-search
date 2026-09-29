"""Unit tests for the shared SqlBackend logic (no database: `_query` is faked)."""

from typing import Any

import pytest

from agentic_search.backends.base import BackendError
from agentic_search.backends.mysql import MySQLBackend
from agentic_search.backends.postgres import PostgresBackend
from agentic_search.backends.sql_backend import SqlBackend


class DistinctFails(SqlBackend):
    dialect = "postgres"
    backend_type = "fake"

    def __init__(self) -> None:
        super().__init__("fake")
        self.queries: list[tuple[str, list[Any] | None]] = []

    async def _query(self, sql: str, params: list[Any] | None) -> list[dict[str, Any]]:
        self.queries.append((sql, params))
        if "DISTINCT" in sql:
            raise BackendError("could not identify an equality operator for type point")
        return []


async def test_samples_is_best_effort():
    b = DistinctFails()
    assert await b._samples("t", "location") is None


async def test_samples_scan_is_capped():
    b = DistinctFails()
    await b._samples("t", "kind")
    [(sql, params)] = b.queries
    assert sql == ('SELECT DISTINCT v FROM (SELECT "kind" AS v FROM "t" WHERE "kind" IS NOT NULL '
                   'LIMIT %s) s LIMIT %s')
    assert params == [10_000, 21]


def _fake_query(answers: dict[str, list[dict[str, Any]]], log: list[str]):
    async def _query(sql: str, params: list[Any] | None) -> list[dict[str, Any]]:
        log.append(sql)
        for needle, rows in answers.items():
            if needle in sql:
                return rows
        return []
    return _query


async def test_postgres_samples_only_mapped_keyword_columns():
    b = PostgresBackend("pg", "postgresql://u:p@h/db")
    log: list[str] = []
    cols = [{"table_name": "t", "column_name": c, "data_type": dt, "udt_name": udt}
            for c, dt, udt in [("id", "text", "text"), ("kind", "character varying", "varchar"),
                               ("loc", "point", "point"), ("doc", "xml", "xml")]]
    b._query = _fake_query({"information_schema.columns": cols,
                            "PRIMARY KEY": [{"table_name": "t", "column_name": "id"}],
                            "DISTINCT": [{"v": "a"}], "COUNT(*)": [{"n": 1}]}, log)
    tables = await b._discover_tables()
    sampled = [s for s in log if "DISTINCT" in s]
    assert len(sampled) == 1 and '"kind"' in sampled[0]
    fields = {f.name: f for f in tables["t"].fields}
    assert fields["kind"].sample_values == ["a"] and fields["loc"].sample_values is None


async def test_mysql_samples_only_mapped_keyword_columns():
    b = MySQLBackend("my", "mysql://u:p@h/db")
    log: list[str] = []
    cols = [{"table_name": "t", "column_name": c, "data_type": dt, "column_type": ct}
            for c, dt, ct in [("id", "varchar", "varchar(20)"), ("kind", "varchar", "varchar(20)"),
                              ("flag", "tinyint", "tinyint(1)"), ("loc", "point", "point")]]
    b._query = _fake_query({"information_schema.COLUMNS": cols,
                            "KEY_COLUMN_USAGE": [{"table_name": "t", "column_name": "id"}],
                            "DISTINCT": [{"v": "a"}], "COUNT(*)": [{"n": 1}]}, log)
    tables = await b._discover_tables()
    sampled = [s for s in log if "DISTINCT" in s]
    assert len(sampled) == 2 and not any("`loc`" in s for s in sampled)
    assert {f.name: f for f in tables["t"].fields}["loc"].sample_values is None


@pytest.mark.parametrize("backend", ["pg", "my"])
async def test_skipped_is_not_duplicated_on_rediscovery(backend):
    if backend == "pg":
        b = PostgresBackend("pg", "postgresql://u:p@h/db")
        cols = [{"table_name": "nopk", "column_name": "x", "data_type": "integer", "udt_name": "int4"}]
        b._query = _fake_query({"information_schema.columns": cols}, [])
    else:
        b = MySQLBackend("my", "mysql://u:p@h/db")
        cols = [{"table_name": "nopk", "column_name": "x", "data_type": "int", "column_type": "int"}]
        b._query = _fake_query({"information_schema.COLUMNS": cols}, [])
    await b._discover_tables()
    await b._discover_tables()
    assert [s.split(" ")[0] for s in b.skipped] == ["nopk"]
