"""BigQuery has no local emulator we rely on: these tests drive the backend with a fake client and
check the SQL, parameters and byte-cap behaviour. `test_live_bigquery` runs against a real dataset."""

import os

import pytest
from google.cloud import bigquery

from agentic_search.backends.base import BackendError
from agentic_search.backends.bigquery import BigQueryBackend
from agentic_search.core.types import (
    Aggregate,
    Capability,
    Eq,
    Fetch,
    FieldType,
    Lexical,
    Native,
    Regex,
    Vector,
)

ROWS = [{"id": "d1", "title": "Aspirin", "body": "Aspirin relieves headache", "type": "drug", "year": 2020}]


class FakeJob:
    def __init__(self, bytes_, rows):
        self.total_bytes_processed = bytes_
        self._rows = rows

    def result(self):
        return [dict(r) for r in self._rows]


class FakeTable:
    def __init__(self, schema, num_rows=5):
        self.schema = schema
        self.num_rows = num_rows
        self.table_constraints = None


class FakeClient:
    def __init__(self, bytes_=1000, rows=ROWS):
        self.bytes_ = bytes_
        self.rows = rows
        self.queries = []

    def list_tables(self, dataset):
        return [type("T", (), {"table_id": "docs"})()]

    def get_table(self, ref):
        S = bigquery.SchemaField
        return FakeTable([S("id", "STRING"), S("title", "STRING"), S("body", "STRING"),
                          S("type", "STRING"), S("year", "INTEGER"),
                          S("embedding", "FLOAT64", mode="REPEATED")])

    def query(self, sql, job_config):
        self.queries.append((sql, job_config.dry_run, {p.name: getattr(p, "value", getattr(p, "values", None))
                                                        for p in job_config.query_parameters}))
        return FakeJob(self.bytes_, [] if job_config.dry_run else self.rows)


def make(client=None, **kw):
    return BigQueryBackend("bq", "proj", "ds", client=client or FakeClient(),
                           text_columns={"docs": ["title", "body"]},
                           vector_dims={"docs.embedding": 64},
                           embedders={"docs.embedding": "hash64"}, **kw)


async def test_discover_maps_schema():
    m = await make(native_query=True).discover()
    coll = m.resolve_collection("docs")
    assert coll.count == 5
    assert coll.field("title").type is FieldType.TEXT and coll.field("type").type is FieldType.KEYWORD
    emb = coll.field("embedding")
    assert emb.type is FieldType.VECTOR and emb.vector_dim == 64 and emb.embedder_id == "hash64"
    assert {Capability.LEXICAL, Capability.VECTOR, Capability.HYBRID, Capability.NATIVE} <= m.capabilities


async def test_lexical_sql_dry_runs_then_runs():
    client = FakeClient()
    b = make(client)
    hits = await b.execute(Lexical(source="bq", text="Headache relief", filter=Eq(field="year", value=2020)))
    assert [h.doc_id for h in hits] == ["d1"] and "headache" in hits[0].content[0].text.lower()
    (dry_sql, dry, params), (sql, run_dry, _) = client.queries
    assert dry is True and not run_dry and dry_sql == sql
    assert "CONTAINS_SUBSTR((`title`, `body`), @p0)" in sql and "`year` = @p2" in sql
    assert "FROM `proj.ds.docs`" in sql and "LIMIT @p3" in sql
    assert params == {"p0": "headache", "p1": "relief", "p2": 2020, "p3": 20}


async def test_byte_cap_refuses_before_running():
    client = FakeClient(bytes_=5_000_000_000)
    with pytest.raises(BackendError, match="would scan"):
        await make(client).execute(Regex(source="bq", pattern="asp"))
    assert len(client.queries) == 1 and client.queries[0][1] is True


async def test_vector_search_sql():
    client = FakeClient()
    await make(client).execute(Vector(source="bq", field="embedding", hyde_text="x", vector=[0.5, 1.0], limit=3))
    sql, _, params = client.queries[1]
    assert "FROM VECTOR_SEARCH((SELECT * FROM `proj.ds.docs`), 'embedding'" in sql
    assert "(SELECT @p0 AS `embedding`), top_k => 3, distance_type => 'COSINE'" in sql
    assert "base.`title` AS `title`" in sql and "`embedding` AS" not in sql.split("FROM")[0]
    assert params == {"p0": [0.5, 1.0]}


async def test_regex_fetch_aggregate_native_sql():
    client = FakeClient(rows=[{"type": "drug", "count": 3}])
    b = make(client, native_query=True)
    await b.execute(Regex(source="bq", pattern="print.*", fields=["body"]))
    assert "REGEXP_CONTAINS(CAST(`body` AS STRING), @p0)" in client.queries[-1][0]
    await b.execute(Fetch(source="bq", doc_ids=["d1", "d2"]))
    assert "CAST(`id` AS STRING) IN (@p0, @p1)" in client.queries[-1][0]
    [agg] = await b.execute(Aggregate(source="bq", group_by=["type"], metrics=["count", "avg:year"]))
    assert "GROUP BY `type`" in client.queries[-1][0] and "AVG(`year`) AS `avg_year`" in client.queries[-1][0]
    assert agg.content[0].data == {"type": "drug", "count": 3}
    await b.execute(Native(source="bq", dialect="bigquery", query="SELECT id FROM `proj.ds.docs`"))
    assert client.queries[-1][0].endswith("LIMIT 20")
    with pytest.raises(BackendError):
        await b.execute(Native(source="bq", dialect="sql", query="DELETE FROM `proj.ds.docs` WHERE TRUE"))


async def test_missing_bigquery_extra_on_discover(monkeypatch):
    """Test that discover() raises BackendError when bigquery module is not available."""
    import sys
    monkeypatch.setitem(sys.modules, "google.cloud.bigquery", None)
    b = BigQueryBackend("bq", "proj", "ds")
    with pytest.raises(BackendError, match="bigquery.*extra"):
        await b.discover()


async def test_client_construction_failure_on_discover(monkeypatch):
    """Test that discover() raises BackendError when client construction fails."""
    def raising_client(*args, **kwargs):
        raise RuntimeError("no creds")

    monkeypatch.setattr("google.cloud.bigquery.Client", raising_client)
    b = BigQueryBackend("bq", "proj", "ds")
    with pytest.raises(BackendError, match="no creds"):
        await b.discover()


async def test_non_google_exceptions_in_query():
    """Test that non-Google exceptions in query() are wrapped as BackendError."""
    class FakeClientRaisingQuery(FakeClient):
        def query(self, sql, job_config):
            if not job_config.dry_run:
                raise RuntimeError("refresh failed")
            return super().query(sql, job_config)

    client = FakeClientRaisingQuery()
    b = make(client)
    with pytest.raises(BackendError, match="refresh failed"):
        await b.execute(Regex(source="bq", pattern="asp"))


async def test_non_google_exceptions_in_discover():
    """Test that non-Google exceptions in list_tables() are wrapped as BackendError."""
    class FakeClientRaisingListTables(FakeClient):
        def list_tables(self, dataset):
            raise RuntimeError("boom")

    client = FakeClientRaisingListTables()
    b = make(client)
    with pytest.raises(BackendError, match="boom"):
        await b.discover()


async def test_client_construction_failure_on_query(monkeypatch):
    """Test that client construction failure during query() is wrapped as BackendError."""
    # Pre-populate discovery with a fake client, then set _client to None
    # and mock Client constructor to raise RuntimeError
    client = FakeClient()
    b = make(client)
    await b.discover()  # Populate _tables with the fake client

    # Now set _client to None and patch Client constructor to fail
    b._client = None
    def raising_client(*args, **kwargs):
        raise RuntimeError("no creds at query time")

    monkeypatch.setattr("google.cloud.bigquery.Client", raising_client)
    with pytest.raises(BackendError, match="no creds at query time"):
        await b.execute(Regex(source="bq", pattern="asp"))


@pytest.mark.live
async def test_live_bigquery():
    project, dataset = os.environ.get("GOOGLE_CLOUD_PROJECT"), os.environ.get("AGENTIC_SEARCH_BQ_DATASET")
    if not (project and dataset):
        pytest.skip("set GOOGLE_CLOUD_PROJECT and AGENTIC_SEARCH_BQ_DATASET (a dataset with a `docs` table)")
    b = BigQueryBackend("bq", project, dataset, tables=["docs"], max_bytes_billed=100_000_000)
    m = await b.discover()
    assert m.resolve_collection("docs") is not None
    hits = await b.execute(Lexical(source="bq", collection="docs", text="headache", limit=5))
    assert isinstance(hits, list)
