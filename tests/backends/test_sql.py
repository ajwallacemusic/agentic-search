import datetime as dt
import decimal

import pytest

from agentic_search.backends.base import BackendError
from agentic_search.backends.sql import (
    Params,
    filter_sql,
    parse_metric,
    quote_ident,
    row_to_hit,
    vector_literal,
    where_clause,
)
from agentic_search.core.types import (
    And,
    Contains,
    Eq,
    Exists,
    In,
    Not,
    Or,
    Range,
    StructuredPart,
    TextPart,
)

COLS = {"type", "year", "title", "body"}


def test_quote_ident():
    assert quote_ident("year", "postgres") == '"year"'
    assert quote_ident("year", "mysql") == "`year`"
    assert quote_ident("year", "bigquery") == "`year`"
    with pytest.raises(BackendError):
        quote_ident('x"; DROP TABLE t; --', "postgres")
    with pytest.raises(BackendError):
        quote_ident("year\n", "postgres")


def test_params_styles():
    p = Params("postgres")
    assert (p.add(1), p.add("a"), p.values) == ("%s", "%s", [1, "a"])
    b = Params("bigquery")
    assert (b.add(1), b.add(2)) == ("@p0", "@p1")


def test_filter_translation_postgres():
    p = Params("postgres")
    f = And(clauses=[Eq(field="type", value="drug"),
                     Or(clauses=[Range(field="year", gte=2020, lt=2022), Not(clause=Exists(field="title"))]),
                     In(field="type", values=["a", "b"]), Contains(field="body", value="50%_off")])
    sql = filter_sql(f, "postgres", p, COLS)
    assert sql == ('("type" = %s AND (("year" >= %s AND "year" < %s) OR (NOT "title" IS NOT NULL)) '
                   'AND "type" IN (%s, %s) AND CAST("body" AS TEXT) ILIKE %s)')
    assert p.values == ["drug", 2020, 2022, "a", "b", "%50\\%\\_off%"]


def test_filter_translation_other_dialects():
    m = Params("mysql")
    assert filter_sql(Contains(field="body", value="x"), "mysql", m, COLS) == \
        "LOWER(CAST(`body` AS CHAR)) LIKE LOWER(%s)"
    b = Params("bigquery")
    assert filter_sql(Contains(field="body", value="x"), "bigquery", b, COLS) == \
        "CONTAINS_SUBSTR(CAST(`body` AS STRING), @p0)"
    assert filter_sql(In(field="type", values=[]), "mysql", Params("mysql"), COLS) == "FALSE"
    assert filter_sql(And(clauses=[]), "mysql", Params("mysql"), COLS) == "TRUE"


def test_filter_rejects_unknown_columns():
    with pytest.raises(BackendError, match="unknown column"):
        filter_sql(Eq(field="nope", value=1), "postgres", Params("postgres"), COLS)


def test_where_clause_and_metrics():
    p = Params("postgres")
    assert where_clause(None, "postgres", p, COLS) == ""
    assert where_clause(Eq(field="year", value=1), "postgres", p, COLS, extra=["x IS NOT NULL"]) == \
        ' WHERE x IS NOT NULL AND "year" = %s'
    assert parse_metric("count", COLS, "mysql") == ("COUNT(*)", "count")
    assert parse_metric("avg:year", COLS, "mysql") == ("AVG(`year`)", "avg_year")
    for bad in ("median:year", "sum:", "sum:nope"):
        with pytest.raises(BackendError):
            parse_metric(bad, COLS, "mysql")


def test_row_to_hit():
    row = {"id": 7, "title": "T", "body": "B", "price": decimal.Decimal("1.5"),
           "at": dt.date(2024, 1, 2)}
    h = row_to_hit(row, source="s", id_column="id", text_columns=["title", "body"], score=0.5)
    assert h.key == "s:7" and h.content == [TextPart(text="T\nB")] and h.raw_score == 0.5
    assert h.metadata == {"id": 7, "price": 1.5, "at": "2024-01-02"}
    s = row_to_hit({"n": 1}, source="s", id_column=None, text_columns=[], fallback_id="native:0")
    assert s.doc_id == "native:0" and isinstance(s.content[0], StructuredPart)


def test_vector_literal():
    assert vector_literal([0.5, 1.0, -2.25]) == "[0.5,1,-2.25]"
