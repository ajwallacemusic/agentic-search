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


def test_sql_rejects_side_effecting_functions():
    """Test that dangerous functions are rejected."""
    # Postgres sleep function
    with pytest.raises(NativeQueryRejected, match="function.*not allowed"):
        guard_sql("SELECT pg_sleep(100)", "postgres", 10)

    # Postgres advisory lock
    with pytest.raises(NativeQueryRejected, match="function.*not allowed"):
        guard_sql("SELECT pg_advisory_lock(1)", "postgres", 10)

    # Postgres set_config
    with pytest.raises(NativeQueryRejected, match="function.*not allowed"):
        guard_sql("SELECT set_config('default_transaction_read_only','off',false)", "postgres", 10)

    # Postgres dblink
    with pytest.raises(NativeQueryRejected, match="function.*not allowed"):
        guard_sql("SELECT dblink_exec('x','delete from t')", "postgres", 10)

    # Postgres large object import
    with pytest.raises(NativeQueryRejected, match="function.*not allowed"):
        guard_sql("SELECT lo_import('/etc/passwd')", "postgres", 10)

    # Postgres sequence manipulation
    with pytest.raises(NativeQueryRejected, match="function.*not allowed"):
        guard_sql("SELECT nextval('s')", "postgres", 10)

    # MySQL GET_LOCK
    with pytest.raises(NativeQueryRejected, match="function.*not allowed"):
        guard_sql("SELECT GET_LOCK('a', 10)", "mysql", 10)

    # MySQL SLEEP
    with pytest.raises(NativeQueryRejected, match="function.*not allowed"):
        guard_sql("SELECT SLEEP(10)", "mysql", 10)

    # Postgres notify
    with pytest.raises(NativeQueryRejected, match="function.*not allowed"):
        guard_sql("SELECT pg_notify('channel','message')", "postgres", 10)

    # But ordinary functions should pass
    assert guard_sql("SELECT count(*), lower(title) FROM docs", "postgres", 50)
    assert guard_sql("SELECT COUNT(*) FROM docs", "mysql", 50)


def test_cypher_guard():
    # Test basic queries wrapped in CALL
    assert guard_cypher("MATCH (n:Drug) RETURN n", 20) == "CALL { MATCH (n:Drug) RETURN n } RETURN * LIMIT 20"
    assert guard_cypher("MATCH (n) RETURN n LIMIT 5;", 20) == "CALL { MATCH (n) RETURN n } RETURN * LIMIT 5"
    assert guard_cypher("MATCH (n) RETURN n LIMIT 500", 20) == "CALL { MATCH (n) RETURN n } RETURN * LIMIT 20"
    assert guard_cypher("MATCH (n) WHERE n.name = 'set' RETURN n.offset", 5).endswith("RETURN * LIMIT 5")
    result = guard_cypher("CALL db.index.fulltext.queryNodes('i', 'x') YIELD node RETURN node", 5)
    assert result  # Just verify it doesn't raise
    for bad in ("MATCH (n) DETACH DELETE n", "MERGE (n:X)", "MATCH (n) SET n.x = 1",
                "CALL dbms.components()", "MATCH (n) RETURN n; MATCH (m) DELETE m",
                "LOAD CSV FROM 'x' AS row RETURN row"):
        with pytest.raises(NativeQueryRejected):
            guard_cypher(bad, 5)


def test_cypher_comment_bypasses():
    """Test that comments cannot be used to hide write keywords."""
    # Line comment bypass attempt
    with pytest.raises(NativeQueryRejected, match="comments are not allowed"):
        guard_cypher("MATCH (n) // it's\nSET n.x=1 //'\nRETURN n", 5)

    # Block comment bypass attempt with quotes
    with pytest.raises(NativeQueryRejected, match="comments are not allowed"):
        guard_cypher("MATCH (n) /* ' */ SET n.x=1 /* ' */ RETURN n", 5)

    # Comment-only query
    with pytest.raises(NativeQueryRejected, match="comments are not allowed"):
        guard_cypher("/* just a comment */", 5)


def test_cypher_union_wrapped():
    """Test that UNION queries are wrapped and capped correctly."""
    result = guard_cypher("MATCH (n) RETURN n UNION SELECT 1", 10)
    assert "CALL {" in result and "} RETURN * LIMIT 10" in result


def test_cypher_parameterized_limit():
    """Test that parameterized LIMIT is handled via wrapping."""
    result = guard_cypher("MATCH (n) RETURN n LIMIT $x", 10)
    assert "CALL {" in result and "} RETURN * LIMIT 10" in result


def test_cypher_backtick_escaping():
    """Test that backtick identifiers use doubling, not backslash escapes."""
    # Backtick bypass: in Cypher, backticks doubled mean literal backtick, no backslash escape
    # This query has a SET statement hidden by a fake escaped backtick
    with pytest.raises(NativeQueryRejected, match="SET"):
        guard_cypher(r"MATCH (n) WITH n AS `a\` SET n.x=1 RETURN n AS `b`", 5)

    # Doubled backticks should work (escaped backtick in identifier name)
    result = guard_cypher("MATCH (n) RETURN n AS `we``ird`", 5)
    assert "CALL {" in result


def test_cypher_unterminated_literals():
    """Test that unterminated strings and backticks are rejected."""
    with pytest.raises(NativeQueryRejected, match="unterminated"):
        guard_cypher("MATCH (n) RETURN n 'abc SET n.x=1", 5)

    with pytest.raises(NativeQueryRejected, match="unterminated"):
        guard_cypher('MATCH (n) RETURN n "abc SET n.x=1', 5)

    with pytest.raises(NativeQueryRejected, match="unterminated"):
        guard_cypher("MATCH (n) RETURN n `abc SET n.x=1", 5)


def test_cypher_procedure_without_return():
    """Test that procedure-only calls get RETURN * added."""
    result = guard_cypher("CALL db.labels() YIELD label", 5)
    assert result == "CALL { CALL db.labels() YIELD label RETURN * } RETURN * LIMIT 5"


def test_opensearch_guard():
    body = guard_opensearch('{"query": {"term": {"type": "drug"}}, "size": 500}', 50)
    assert body == {"query": {"term": {"type": "drug"}}, "size": 50}
    assert guard_opensearch('{"query": {"match_all": {}}}', 50)["size"] == 50
    for bad in ('[1]', 'not json', '{"script": {}}', '{"query": {"script_score": {}}}',
                '{"query": {}, "index": "x"}'):
        with pytest.raises(NativeQueryRejected):
            guard_opensearch(bad, 50)


def test_opensearch_size_validation():
    """Test strict validation of size parameter."""
    # Negative size should be rejected or capped to max_rows
    body = guard_opensearch('{"query": {}, "size": -1}', 50)
    assert body["size"] == 50

    # Boolean size should be rejected or capped to max_rows
    body = guard_opensearch('{"query": {}, "size": true}', 50)
    assert body["size"] == 50

    # Size within bounds should be preserved
    body = guard_opensearch('{"query": {}, "size": 30}', 50)
    assert body["size"] == 30

    # No size should default to max_rows
    body = guard_opensearch('{"query": {}}', 50)
    assert body["size"] == 50


def test_opensearch_from_validation():
    """Test validation of from parameter to prevent deep pagination DoS."""
    # Negative from should be rejected
    with pytest.raises(NativeQueryRejected, match="non-negative"):
        guard_opensearch('{"query": {}, "from": -1}', 50)

    # Non-integer from should be rejected (including bool)
    with pytest.raises(NativeQueryRejected, match="non-negative"):
        guard_opensearch('{"query": {}, "from": "abc"}', 50)

    # Boolean from should be rejected
    with pytest.raises(NativeQueryRejected, match="non-negative"):
        guard_opensearch('{"query": {}, "from": true}', 50)

    # Large from + size should be rejected (from + size <= 10000)
    with pytest.raises(NativeQueryRejected, match="exceeds maximum"):
        guard_opensearch('{"query": {}, "from": 9999, "size": 100}', 50)

    # Valid from should pass
    body = guard_opensearch('{"query": {}, "from": 100, "size": 50}', 100)
    assert body["from"] == 100
    assert body["size"] == 50


def test_sql_output_drops_comments():
    out = guard_sql("SELECT id /* hi */ FROM docs -- trailing", "postgres", 10)
    assert "/*" not in out and "hi" not in out and "trailing" not in out


@pytest.mark.parametrize("body", [
    '{"query": {"terms": {"id": {"index": "secrets", "id": "1", "path": "ids"}}}}',
    '{"query": {"more_like_this": {"fields": ["body"], "like": [{"_index": "other", "_id": "1"}]}}}',
    '{"query": {"percolate": {"field": "q", "index": "other", "id": "1"}}}',
])
def test_opensearch_rejects_cross_index_reads(body):
    with pytest.raises(NativeQueryRejected, match="index"):
        guard_opensearch(body, 10)


@pytest.mark.parametrize("bad", [
    "CALL `dbms`.`listConfig`() YIELD name, value RETURN name, value",
    "CALL `dbms.listConfig`() YIELD name RETURN name",
    "CALL db.`labels`() YIELD label RETURN label",
    "MATCH (n) CALL `apoc`.`create`.`node`(['x'], {}) YIELD node RETURN node",
])
def test_cypher_rejects_backtick_quoted_procedures(bad):
    with pytest.raises(NativeQueryRejected, match="CALL"):
        guard_cypher(bad, 5)


@pytest.mark.parametrize("bad", [
    "MATCH (n:docs) RETURN n.id AS id } RETURN id UNION ALL CALL { MATCH (n:docs) RETURN n.id AS id",
    "MATCH (n) RETURN n }",
    "MATCH (n) WHERE n.x = { RETURN n",
])
def test_cypher_rejects_unbalanced_braces(bad):
    with pytest.raises(NativeQueryRejected, match="braces"):
        guard_cypher(bad, 2)


def test_cypher_allows_balanced_subquery_and_braces_in_strings():
    q = "MATCH (n) CALL { WITH n RETURN n.x AS x } RETURN n, x, '}' AS s, {a: 1} AS m"
    assert guard_cypher(q, 3) == f"CALL {{ {q} }} RETURN * LIMIT 3"


def test_cypher_unicode_escape_bypasses():
    # Neo4j decodes backslash-u XXXX escapes before tokenising, so escapes can hide CALL,
    # braces, quotes or write keywords from the guard.
    u = "\\" + "u"
    for bad in [
        f"{u}0043ALL dbms.listConfig() YIELD name, value RETURN name, value",
        (f"UNWIND range(1,10) AS id RETURN id {u}007d RETURN id UNION ALL {u}0043ALL {u}007b "
         "UNWIND range(1,10) AS id RETURN id"),
        f"MATCH (n) WHERE n.x = 'x{u}0027' SET n.y = 1 RETURN n",
        f"MATCH (n) {u}0043REATE (m) RETURN n",
        f"MATCH (n) WHERE n.x = '{u.upper()}0043' RETURN n",
        f"MATCH (n) WHERE n.x = '{u}u0043' RETURN n",
    ]:
        assert bad.count("\\") >= 1
        with pytest.raises(NativeQueryRejected, match="unicode escapes"):
            guard_cypher(bad, 5)
