import sys

import pytest

from agentic_search.backends.base import BackendError, UnsupportedOperation
from agentic_search.backends.milvus import MilvusBackend, filter_expr
from agentic_search.core.secrets import scrub
from agentic_search.core.types import (
    Aggregate,
    And,
    Contains,
    Eq,
    Exists,
    In,
    Not,
    Or,
    Range,
    Regex,
)

FIELDS = {"type", "year", "title", "id"}


def test_filter_expr_templates_every_value():
    params: dict = {}
    expr = filter_expr(And(clauses=[Eq(field="type", value="drug"),
                                    Or(clauses=[Range(field="year", gte=2020, lt=2022),
                                                Not(clause=In(field="id", values=["a", "b"]))]),
                                    Contains(field="title", value='50%_off"x')]), FIELDS, params)
    # LIKE only accepts a literal: wildcards are LIKE-escaped, then the literal is quote-escaped.
    assert expr == ('(type == {p0} and ((year >= {p1} and year < {p2}) or (not (id in {p3}))) '
                    'and title like "%50\\\\%\\\\_off\\"x%")')
    assert params == {"p0": "drug", "p1": 2020, "p2": 2022, "p3": ["a", "b"]}


def test_filter_expr_edge_cases():
    assert filter_expr(And(clauses=[]), FIELDS, {}) == "true"
    assert filter_expr(Or(clauses=[]), FIELDS, {}) == "false"
    assert filter_expr(In(field="type", values=[]), FIELDS, {}) == "false"
    assert filter_expr(Range(field="year"), FIELDS, {}) == "true"
    assert filter_expr(Exists(field="title"), FIELDS, {}) == "title is not null"


@pytest.mark.parametrize("bad", ["nope", "year) or (1 == 1", "type\n"])
def test_filter_expr_rejects_unknown_or_bad_fields(bad):
    with pytest.raises(BackendError):
        filter_expr(Eq(field=bad, value=1), FIELDS | {"year) or (1 == 1", "type\n"}, {})


def test_constructor_registers_secrets():
    MilvusBackend("mv", "http://root:milvus-pw-123@h:19530", token="tok-abcdef-999")
    assert scrub("x milvus-pw-123 tok-abcdef-999") == "x *** ***"


async def test_missing_extra_is_backend_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymilvus", None)
    with pytest.raises(BackendError, match="milvus.*extra"):
        await MilvusBackend("mv", "http://localhost:59530").discover()


async def test_unreachable_uri_is_backend_error():
    b = MilvusBackend("mv", "http://127.0.0.1:1", connect_timeout_s=2)
    try:
        with pytest.raises(BackendError):
            await b.discover()
    finally:
        await b.close()


async def test_unsupported_ops_without_connecting():
    b = MilvusBackend("mv", "http://127.0.0.1:1", connect_timeout_s=2)
    b._collections = {}  # pretend discovery ran
    for op in (Regex(source="mv", pattern="x"), Aggregate(source="mv", group_by=["type"])):
        with pytest.raises(UnsupportedOperation):
            await b.execute(op)
