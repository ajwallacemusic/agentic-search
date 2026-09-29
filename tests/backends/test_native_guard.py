import pytest

from agentic_search.backends.native_guard import (
    NativeQueryRejected,
    guard_cypher,
    guard_opensearch,
    guard_sql,
)

WRITES = [
    "DELETE FROM docs",
    "UPDATE docs SET title = 'x'",
    "INSERT INTO docs (id) VALUES ('z')",
    "DROP TABLE docs",
    "CREATE TABLE t (x int)",
    "SELECT 1; DELETE FROM docs",
    "WITH d AS (DELETE FROM docs RETURNING *) SELECT * FROM d",
    "SELECT * INTO copy FROM docs",
    "SELECT * FROM docs FOR UPDATE",
    "TRUNCATE docs",
]


@pytest.mark.parametrize("query", WRITES)
@pytest.mark.parametrize("dialect", ["postgres", "mysql"])
def test_sql_rejects_writes(query, dialect):
    if dialect == "mysql" and query.startswith(("WITH d AS", "SELECT * INTO")):
        pytest.skip("postgres-only syntax")
    with pytest.raises(NativeQueryRejected):
        guard_sql(query, dialect, 100)


def test_sql_limits():
    assert guard_sql("SELECT id FROM docs", "postgres", 50) == "SELECT id FROM docs LIMIT 50"
    assert guard_sql("SELECT id FROM docs LIMIT 5", "mysql", 50) == "SELECT id FROM docs LIMIT 5"
    assert guard_sql("SELECT id FROM docs LIMIT 500", "postgres", 50) == "SELECT id FROM docs LIMIT 50"
    assert "LIMIT 10" in guard_sql("SELECT a FROM x UNION ALL SELECT a FROM y", "bigquery", 10)
    with pytest.raises(NativeQueryRejected, match="parse"):
        guard_sql("SELEC nonsense((", "postgres", 10)


def test_cypher_guard():
    assert guard_cypher("MATCH (n:Drug) RETURN n", 20) == "MATCH (n:Drug) RETURN n LIMIT 20"
    assert guard_cypher("MATCH (n) RETURN n LIMIT 5;", 20) == "MATCH (n) RETURN n LIMIT 5"
    assert guard_cypher("MATCH (n) RETURN n LIMIT 500", 20) == "MATCH (n) RETURN n LIMIT 20"
    assert guard_cypher("MATCH (n) WHERE n.name = 'set' RETURN n.offset", 5).endswith("LIMIT 5")
    assert guard_cypher("CALL db.index.fulltext.queryNodes('i', 'x') YIELD node RETURN node", 5)
    for bad in ("MATCH (n) DETACH DELETE n", "MERGE (n:X)", "MATCH (n) SET n.x = 1",
                "CALL dbms.components()", "MATCH (n) RETURN n; MATCH (m) DELETE m",
                "LOAD CSV FROM 'x' AS row RETURN row"):
        with pytest.raises(NativeQueryRejected):
            guard_cypher(bad, 5)


def test_opensearch_guard():
    body = guard_opensearch('{"query": {"term": {"type": "drug"}}, "size": 500}', 50)
    assert body == {"query": {"term": {"type": "drug"}}, "size": 50}
    assert guard_opensearch('{"query": {"match_all": {}}}', 50)["size"] == 50
    for bad in ('[1]', 'not json', '{"script": {}}', '{"query": {"script_score": {}}}',
                '{"query": {}, "index": "x"}'):
        with pytest.raises(NativeQueryRejected):
            guard_opensearch(bad, 50)
