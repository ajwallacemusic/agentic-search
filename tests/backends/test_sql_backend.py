"""Unit tests for the shared SqlBackend logic (no database: `_query` is faked)."""

from typing import Any

import pytest

from agentic_search.backends.base import BackendError
from agentic_search.backends.bigquery import BigQueryBackend
from agentic_search.backends.mysql import MySQLBackend
from agentic_search.backends.native_guard import NativeQueryRejected
from agentic_search.backends.postgres import PostgresBackend
from agentic_search.backends.sql_backend import SqlBackend, TableInfo, field_flags
from agentic_search.core.types import FieldSpec, FieldType, Native


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


@pytest.mark.parametrize("backend", ["pg", "my"])
async def test_composite_primary_key_is_skipped_unless_id_column_given(backend):
    def make(**kw):
        if backend == "pg":
            b = PostgresBackend("pg", "postgresql://u:p@h/db", **kw)
            cols = [{"table_name": "pairs", "column_name": c, "data_type": "text", "udt_name": "text"}
                    for c in ("a", "b")]
            b._query = _fake_query({"information_schema.columns": cols, "PRIMARY KEY": pks}, [])
        else:
            b = MySQLBackend("my", "mysql://u:p@h/db", **kw)
            cols = [{"table_name": "pairs", "column_name": c, "data_type": "text", "column_type": "text"}
                    for c in ("a", "b")]
            b._query = _fake_query({"information_schema.COLUMNS": cols, "KEY_COLUMN_USAGE": pks}, [])
        return b

    pks = [{"table_name": "pairs", "column_name": "a"}, {"table_name": "pairs", "column_name": "b"}]
    b = make()
    assert await b._discover_tables() == {}
    assert b.skipped == ["pairs (composite primary key; set id_columns)"]
    b = make(id_columns={"pairs": "b"})
    assert (await b._discover_tables())["pairs"].id_column == "b"


async def test_aggregate_needs_a_metric():
    from agentic_search.backends.sql_backend import TableInfo
    from agentic_search.core.types import Aggregate, FieldSpec, FieldType

    b = DistinctFails()
    b._tables = {"t": TableInfo(name="t", id_column="id",
                                fields=[FieldSpec(name="id", type=FieldType.KEYWORD),
                                        FieldSpec(name="kind", type=FieldType.KEYWORD)])}
    with pytest.raises(BackendError, match="at least one metric"):
        await b.execute(Aggregate(source="fake", group_by=["kind"], metrics=[]))


def _text(name: str) -> FieldSpec:
    return FieldSpec(name=name, type=FieldType.TEXT, **field_flags(FieldType.TEXT))


class TwoTables(SqlBackend):
    dialect = "postgres"
    backend_type = "fake"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("fake", native_query=True, **kwargs)
        self.queries: list[str] = []

    async def _discover_tables(self) -> dict[str, TableInfo]:
        return {"visits": TableInfo("visits", "id", [_text("id"), _text("dx"), _text("ssn")]),
                "people": TableInfo("people", "id", [_text("id"), _text("name")])}

    def _native_db(self) -> str | None:
        return "public"

    async def _query(self, sql: str, params: list[Any] | None) -> list[dict[str, Any]]:
        self.queries.append(sql)
        return [{"id": "v1", "dx": "pain"}]


async def test_columns_trim_discovery_and_keep_the_id():
    b = TwoTables(columns={"visits": ["dx"]})
    manifest = await b.discover()
    [visits] = manifest.collections
    assert visits.name == "visits"
    assert [f.name for f in visits.fields] == ["id", "dx"]


async def test_no_columns_keeps_everything():
    b = TwoTables()
    manifest = await b.discover()
    assert {c.name for c in manifest.collections} == {"visits", "people"}


async def test_native_sql_runs_the_resolved_query():
    b = TwoTables(columns={"visits": ["dx"]})
    hits = await b.execute(Native(source="fake", collection="visits", dialect="sql",
                                  query="SELECT id, dx FROM visits"))
    assert b.queries == ["SELECT visits.id AS id, visits.dx AS dx FROM public.visits AS visits LIMIT 20"]
    assert hits[0].doc_id


@pytest.mark.parametrize("query", ["SELECT ssn FROM visits", "SELECT COUNT(*) FROM people"])
async def test_native_sql_refuses_what_discovery_hid(query):
    b = TwoTables(columns={"visits": ["dx"]})
    with pytest.raises(NativeQueryRejected):
        await b.execute(Native(source="fake", collection="visits", dialect="sql", query=query))


async def test_native_functions_narrow_the_default_list():
    b = TwoTables(columns={"visits": ["dx"]}, native_functions=["count"])
    await b.execute(Native(source="fake", dialect="sql", query="SELECT COUNT(*) FROM visits"))
    with pytest.raises(NativeQueryRejected, match="LOWER"):
        await b.execute(Native(source="fake", dialect="sql", query="SELECT LOWER(dx) FROM visits"))


def test_qualifiers_per_backend():
    bq = BigQueryBackend("bq", "p-1", "ds", client=object())
    assert (bq._native_catalog(), bq._native_db()) == ("p-1", "ds")
    pg = PostgresBackend("pg", "postgresql://u:p@h/db", schema="clinic")
    assert (pg._native_catalog(), pg._native_db()) == (None, "clinic")
