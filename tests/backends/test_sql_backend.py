"""Unit tests for the shared SqlBackend logic (no database: `_query` is faked)."""

from typing import Any

import pytest

from agentic_search.backends.base import BackendError
from agentic_search.backends.bigquery import BigQueryBackend
from agentic_search.backends.mysql import MySQLBackend
from agentic_search.backends.native_guard import NativeQueryRejected
from agentic_search.backends.postgres import PostgresBackend
from agentic_search.backends.sql_backend import SqlBackend, TableInfo, field_flags
from agentic_search.core.types import (
    Aggregate,
    Eq,
    Fetch,
    FieldSpec,
    FieldType,
    FilterOnly,
    Lexical,
    Native,
    Regex,
    Vector,
)


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


class FakePostgres(PostgresBackend):
    """Postgres with a stored tsvector `search_vec` that indexes `dx` and the hidden `ssn`."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("pg", "postgresql://u:p@h/db", **kwargs)
        self.queries: list[str] = []

    async def _discover_tables(self) -> dict[str, TableInfo]:
        self._tsv = {"visits": "search_vec", "gone": "search_vec"}
        self.skipped = ["gone (no primary key / id column)", "kept (composite primary key; set id_columns)"]
        return {"visits": TableInfo("visits", "id", [_text("id"), _text("dx"), _text("ssn")])}

    async def _query(self, sql: str, params: list[Any] | None) -> list[dict[str, Any]]:
        self.queries.append(sql)
        return []


async def test_unlisted_stored_tsvector_is_not_used():
    b = FakePostgres(columns={"visits": ["dx"]})
    await b.execute(Lexical(source="pg", collection="visits", text="pain"))
    [sql] = b.queries
    assert "search_vec" not in sql
    assert "ssn" not in sql
    assert '"dx"' in sql


async def test_listed_stored_tsvector_is_still_used():
    b = FakePostgres(columns={"visits": ["dx", "search_vec"]})
    await b.execute(Lexical(source="pg", collection="visits", text="pain"))
    [sql] = b.queries
    assert '"search_vec"' in sql
    assert "ssn" not in sql


async def test_no_columns_keeps_the_stored_tsvector():
    b = FakePostgres()
    await b.execute(Lexical(source="pg", collection="visits", text="pain"))
    assert '"search_vec"' in b.queries[0]


async def test_skipped_names_only_tables_that_survive_the_restriction():
    b = FakePostgres(columns={"visits": ["dx"], "kept": ["x"]})
    manifest = await b.discover()
    assert manifest.description == "Skipped tables: kept (composite primary key; set id_columns)."
    assert "gone" not in (manifest.description or "")


async def test_native_functions_without_columns_is_an_error():
    with pytest.raises(ValueError, match="native_functions"):
        TwoTables(native_functions=["COUNT"])


async def test_empty_native_functions_allows_none():
    b = TwoTables(columns={"visits": ["dx"]}, native_functions=[])
    with pytest.raises(NativeQueryRejected, match="COUNT"):
        await b.execute(Native(source="fake", dialect="sql", query="SELECT COUNT(*) FROM visits"))
    await b.execute(Native(source="fake", dialect="sql", query="SELECT dx FROM visits"))


class HiddenEverywhere(PostgresBackend):
    """`ssn` is hidden and `emb` is a hidden vector column."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("fake", "postgresql://u:p@h/db", **kwargs)
        self.queries: list[str] = []

    async def _discover_tables(self) -> dict[str, TableInfo]:
        emb = FieldSpec(name="emb", type=FieldType.VECTOR, vector_dim=3, vector_metric="cosine")
        return {"visits": TableInfo("visits", "id", [_text("id"), _text("dx"), _text("ssn"), emb])}

    async def _query(self, sql: str, params: list[Any] | None) -> list[dict[str, Any]]:
        self.queries.append(sql)
        return []


HIDDEN_OPS = [
    FilterOnly(source="fake", collection="visits", filter=Eq(field="ssn", value="1")),
    Regex(source="fake", collection="visits", pattern="1", fields=["ssn"]),
    Regex(source="fake", collection="visits", pattern="1", filter=Eq(field="ssn", value="1")),
    Aggregate(source="fake", collection="visits", group_by=["ssn"]),
    Aggregate(source="fake", collection="visits", group_by=["dx"], metrics=["max:ssn"]),
    Lexical(source="fake", collection="visits", text="pain", fields=["ssn"]),
    Lexical(source="fake", collection="visits", text="pain", filter=Eq(field="ssn", value="1")),
    Vector(source="fake", collection="visits", field="emb", hyde_text="x", vector=[0.1, 0.2, 0.3]),
]


@pytest.mark.parametrize("op", HIDDEN_OPS, ids=lambda op: op.type)
async def test_ops_refuse_a_hidden_column(op):
    b = HiddenEverywhere(columns={"visits": ["dx"]})
    with pytest.raises(BackendError):
        await b.execute(op)
    assert b.queries == []


async def test_fetch_selects_only_allowed_columns():
    b = HiddenEverywhere(columns={"visits": ["dx"]})
    await b.execute(Fetch(source="fake", collection="visits", doc_ids=["v1"]))
    [sql] = b.queries
    assert sql.startswith('SELECT "id", "dx" FROM')
    assert "ssn" not in sql
    assert "emb" not in sql


async def test_mysql_keeps_only_fulltext_indexes_inside_the_allowed_columns():
    class FakeMySQL(MySQLBackend):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__("my", "mysql://u:p@h/db", **kwargs)
            self.queries: list[str] = []

        async def _discover_tables(self) -> dict[str, TableInfo]:
            fields = [_text(n) for n in ("id", "title", "body", "secret")]
            return {"docs": TableInfo("docs", "id", fields, fulltext=[["body", "secret"], ["title"]])}

        async def _query(self, sql: str, params: list[Any] | None) -> list[dict[str, Any]]:
            self.queries.append(sql)
            return []

    b = FakeMySQL(columns={"docs": ["title", "body"]})
    [docs] = (await b.discover()).collections
    assert [f.name for f in docs.fields] == ["id", "title", "body"]
    await b.execute(Lexical(source="my", collection="docs", text="pain"))
    assert "MATCH(`title`)" in b.queries[0]
    assert "secret" not in b.queries[0]
    with pytest.raises(BackendError, match="not searchable"):
        await b.execute(Lexical(source="my", collection="docs", text="pain", fields=["body", "secret"]))
    with pytest.raises(BackendError, match="no FULLTEXT index on exactly"):
        await b.execute(Lexical(source="my", collection="docs", text="pain", fields=["body"]))


async def test_bigquery_camel_case_columns_through_the_backend():
    class FakeBigQuery(BigQueryBackend):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__("bq", "p-1", "ds", client=object(), **kwargs)
            self.queries: list[str] = []

        async def _discover_tables(self) -> dict[str, TableInfo]:
            fields = [_text(n) for n in ("Id", "StudyDescription", "PatientName")]
            return {"Studies": TableInfo("Studies", "Id", fields)}

        async def _query(self, sql: str, params: list[Any] | None) -> list[dict[str, Any]]:
            self.queries.append(sql)
            return []

    b = FakeBigQuery(native_query=True, columns={"Studies": ["StudyDescription"]})
    await b.execute(Native(source="bq", collection="Studies", dialect="sql",
                           query="SELECT StudyDescription FROM Studies"))
    await b.execute(Native(source="bq", collection="Studies", dialect="sql", query="SELECT * FROM Studies"))
    assert all("PatientName" not in q and "patientname" not in q for q in b.queries)
    assert all("`p-1`.ds.Studies" in q or "p-1.ds.Studies" in q for q in b.queries)
    with pytest.raises(NativeQueryRejected):
        await b.execute(Native(source="bq", collection="Studies", dialect="sql",
                               query="SELECT PatientName FROM Studies"))
    with pytest.raises(NativeQueryRejected):
        await b.execute(Native(source="bq", collection="Studies", dialect="sql",
                               query="SELECT StudyDescription FROM studies"))


class OutputNamedBigQuery(BigQueryBackend):
    """Answers each native query with one row keyed by the query's own output names, the way
    BigQuery names result fields."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("bq", "p-1", "ds", client=object(), native_query=True, **kwargs)
        self.queries: list[str] = []

    async def _discover_tables(self) -> dict[str, TableInfo]:
        fields = [_text(n) for n in ("Id", "Modality", "PatientName")]
        return {"Studies": TableInfo("Studies", "Id", fields)}

    async def _query(self, sql: str, params: list[Any] | None) -> list[dict[str, Any]]:
        import sqlglot

        self.queries.append(sql)
        names = sqlglot.parse_one(sql, read="bigquery").named_selects
        return [{name: "s1" if name.lower() == "id" else "CT" for name in names}]


@pytest.mark.parametrize("query", [
    "SELECT Modality, Id FROM Studies",
    "SELECT modality, id FROM Studies",
    "SELECT * FROM Studies",
    "SELECT s.Modality, s.Id FROM Studies s",
    "SELECT x.Modality, x.Id FROM (SELECT Modality, Id FROM Studies) x",
])
async def test_bigquery_native_rows_keep_stored_column_names(query):
    b = OutputNamedBigQuery(columns={"Studies": ["Modality"]})
    [hit] = await b.execute(Native(source="bq", collection="Studies", dialect="sql", query=query))
    assert hit.doc_id == "s1"
    assert hit.metadata == {"Id": "s1", "Modality": "CT"}


async def test_bigquery_native_rows_keep_an_explicit_alias():
    b = OutputNamedBigQuery(columns={"Studies": ["Modality"]})
    [hit] = await b.execute(Native(source="bq", collection="Studies", dialect="sql",
                                   query="SELECT Id, Modality AS ScanKind, COUNT(*) AS N FROM Studies "
                                         "GROUP BY Id, Modality ORDER BY n DESC"))
    assert hit.doc_id == "s1"
    assert set(hit.metadata) == {"Id", "ScanKind", "N"}
    assert "patientname" not in b.queries[0].lower()
