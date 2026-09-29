"""Shared SQL building blocks: identifier quoting, parameter collection, Filter AST → WHERE clause,
and row → Hit conversion for the Postgres, MySQL and BigQuery adapters."""

from __future__ import annotations

import datetime as dt
import decimal
import json
import re
import uuid
from typing import Any, Literal

from agentic_search.backends.base import BackendError
from agentic_search.core.types import (
    And,
    Contains,
    Content,
    Eq,
    Exists,
    Filter,
    Hit,
    In,
    Not,
    Or,
    Range,
    StructuredPart,
    TextPart,
)

Dialect = Literal["postgres", "mysql", "bigquery"]
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
AGG_FUNCS = {"sum": "SUM", "avg": "AVG", "min": "MIN", "max": "MAX"}


def quote_ident(name: str, dialect: Dialect) -> str:
    """Quote a column/table name after checking it is a plain identifier (no injection surface)."""
    if not _IDENT.fullmatch(name):
        raise BackendError(f"invalid identifier {name!r}")
    if dialect == "postgres":
        return f'"{name}"'
    return f"`{name}`"


class Params:
    """Collects bind values in placeholder order: %s for postgres/mysql, @pN for bigquery."""

    def __init__(self, dialect: Dialect):
        self.dialect = dialect
        self.values: list[Any] = []

    def add(self, value: Any) -> str:
        self.values.append(value)
        return f"@p{len(self.values) - 1}" if self.dialect == "bigquery" else "%s"


def _like_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def filter_sql(f: Filter, dialect: Dialect, params: Params, columns: set[str]) -> str:
    """Translate a filter to a parameterized boolean SQL expression over known columns."""
    if isinstance(f, And):
        return "(" + " AND ".join(filter_sql(c, dialect, params, columns) for c in f.clauses) + ")" \
            if f.clauses else "TRUE"
    if isinstance(f, Or):
        return "(" + " OR ".join(filter_sql(c, dialect, params, columns) for c in f.clauses) + ")" \
            if f.clauses else "FALSE"
    if isinstance(f, Not):
        return f"(NOT {filter_sql(f.clause, dialect, params, columns)})"
    if f.field not in columns:
        raise BackendError(f"unknown column {f.field!r}")
    col = quote_ident(f.field, dialect)
    if isinstance(f, Eq):
        return f"{col} = {params.add(f.value)}"
    if isinstance(f, In):
        if not f.values:
            return "FALSE"
        return f"{col} IN ({', '.join(params.add(v) for v in f.values)})"
    if isinstance(f, Range):
        parts = [f"{col} {sym} {params.add(bound)}" for bound, sym in
                 ((f.gte, ">="), (f.gt, ">"), (f.lte, "<="), (f.lt, "<")) if bound is not None]
        return "(" + " AND ".join(parts) + ")" if parts else "TRUE"
    if isinstance(f, Exists):
        return f"{col} IS NOT NULL"
    if isinstance(f, Contains):
        if dialect == "bigquery":
            return f"CONTAINS_SUBSTR(CAST({col} AS STRING), {params.add(f.value)})"
        pattern = params.add(f"%{_like_escape(f.value)}%")
        if dialect == "postgres":
            return f"CAST({col} AS TEXT) ILIKE {pattern}"
        return f"LOWER(CAST({col} AS CHAR)) LIKE LOWER({pattern})"
    raise BackendError(f"unsupported filter node {type(f).__name__}")


def where_clause(f: Filter | None, dialect: Dialect, params: Params, columns: set[str],
                 extra: list[str] | None = None) -> str:
    parts = list(extra or [])
    if f is not None:
        parts.append(filter_sql(f, dialect, params, columns))
    return " WHERE " + " AND ".join(parts) if parts else ""


def parse_metric(metric: str, columns: set[str], dialect: Dialect) -> tuple[str, str]:
    """'count' → COUNT(*); 'sum:price' → SUM(`price`). Returns (sql_expr, output_name)."""
    if metric == "count":
        return "COUNT(*)", "count"
    fn, _, col = metric.partition(":")
    if fn not in AGG_FUNCS or not col:
        raise BackendError(f"unsupported metric {metric!r}; use count or sum|avg|min|max:<column>")
    if col not in columns:
        raise BackendError(f"unknown column {col!r} in metric {metric!r}")
    return f"{AGG_FUNCS[fn]}({quote_ident(col, dialect)})", f"{fn}_{col}"


def jsonable(value: Any) -> Any:
    """Make DB values JSON-friendly for Hit metadata / StructuredPart."""
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    return value


def row_to_hit(row: dict[str, Any], *, source: str, id_column: str | None, text_columns: list[str],
               score: float | None = None, fallback_id: str = "") -> Hit:
    """TEXT columns become the hit's text content; everything else becomes metadata."""
    clean = {k: jsonable(v) for k, v in row.items()}
    doc_id = str(clean.get(id_column)) if id_column and clean.get(id_column) is not None else fallback_id
    texts = [str(clean[c]) for c in text_columns if clean.get(c) not in (None, "")]
    content: list[Content] = [TextPart(text="\n".join(texts))] if texts else [StructuredPart(data=clean)]
    metadata = {k: v for k, v in clean.items() if k not in text_columns}
    return Hit(doc_id=doc_id, source=source, content=content, metadata=metadata, raw_score=score)


def vector_literal(vector: list[float]) -> str:
    """pgvector text form: '[0.1,0.2,…]'."""
    return "[" + ",".join(f"{x:.8g}" for x in vector) + "]"


def dumps(value: Any) -> str:
    return json.dumps(jsonable(value), sort_keys=True, default=str)
