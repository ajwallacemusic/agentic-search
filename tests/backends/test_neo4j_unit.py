import sys

import pytest

from agentic_search.backends.base import BackendError
from agentic_search.core.secrets import scrub
from agentic_search.core.types import (
    And,
    Contains,
    Eq,
    Exists,
    In,
    Not,
    Or,
    Range,
    Traverse,
)

PROPS = {"type", "year", "title"}


def test_quote_name_doubles_backticks():
    from agentic_search.backends.neo4j import quote_name
    assert quote_name("docs") == "`docs`"
    assert quote_name("a`b") == "`a``b`"


def test_filter_cypher_translation():
    from agentic_search.backends.neo4j import filter_cypher
    params: dict = {}
    f = And(clauses=[Eq(field="type", value="drug"),
                     Or(clauses=[Range(field="year", gte=2020, lt=2022), Not(clause=Exists(field="title"))]),
                     In(field="type", values=["a", "b"]), Contains(field="title", value="Asp")])
    cypher = filter_cypher(f, params, PROPS, "n")
    assert cypher == ("(n.`type` = $p0 AND ((n.`year` >= $p1 AND n.`year` < $p2) OR (NOT n.`title` IS NOT NULL)) "
                      "AND n.`type` IN $p3 AND toLower(toString(n.`title`)) CONTAINS toLower($p4))")
    assert params == {"p0": "drug", "p1": 2020, "p2": 2022, "p3": ["a", "b"], "p4": "Asp"}
    assert filter_cypher(And(clauses=[]), {}, PROPS, "n") == "true"
    assert filter_cypher(In(field="type", values=[]), {}, PROPS, "n") == "false"


def test_filter_cypher_rejects_unknown_property():
    from agentic_search.backends.neo4j import filter_cypher
    with pytest.raises(BackendError, match="unknown property"):
        filter_cypher(Eq(field="x`) DETACH DELETE n //", value=1), {}, PROPS, "n")


def test_lucene_query_is_or_of_lowercase_terms():
    from agentic_search.backends.neo4j import lucene_query
    assert lucene_query("Headache AND pain!") == "headache OR and OR pain"
    with pytest.raises(BackendError):
        lucene_query("!!!")


async def test_traverse_validates_identifiers(monkeypatch):
    from agentic_search.backends import neo4j as mod

    b = mod.Neo4jBackend("neo", "bolt://localhost:1")
    schema = mod._Schema(labels={"docs": mod._Label(name="docs", properties={"id": None}),
                                 "conditions": mod._Label(name="conditions", properties={"id": None})},
                         relationship_types=["TREATS"])

    async def fake_schema():
        return schema

    monkeypatch.setattr(b, "_ensure_schema", fake_schema)
    for bad in (dict(rel_types=["TREATS`]->(x) DELETE x //"]), dict(target_label="nope"),
                dict(collection="nope")):
        kwargs = {"rel_types": ["TREATS"], **bad}
        with pytest.raises(BackendError):
            await b.execute(Traverse(source="neo", start=Eq(field="id", value="h"), **kwargs))


async def test_missing_extra_is_backend_error(monkeypatch):
    from agentic_search.backends.neo4j import Neo4jBackend
    monkeypatch.setitem(sys.modules, "neo4j", None)
    with pytest.raises(BackendError, match="neo4j.*extra"):
        await Neo4jBackend("neo", "bolt://localhost:57687").discover()


async def test_unreachable_is_backend_error_and_password_registered():
    from agentic_search.backends.neo4j import Neo4jBackend
    b = Neo4jBackend("neo", "bolt://127.0.0.1:1", user="neo4j", password="secretpw3", connect_timeout_s=2)
    try:
        with pytest.raises(BackendError) as excinfo:
            await b.discover()
    finally:
        await b.close()
    assert scrub("leak secretpw3 here") == "leak *** here"
    assert "secretpw3" not in scrub(str(excinfo.value))
