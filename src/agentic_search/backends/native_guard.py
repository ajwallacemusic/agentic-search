"""Read-only gate for `native_query`: SQL (via sqlglot), Cypher (keyword scan), OpenSearch DSL.

Every guard either returns a query that is safe to run (with a row cap applied) or raises
NativeQueryRejected. Adapters still connect with read-only sessions; this is the first line."""

from __future__ import annotations

import json
import re
from typing import Any

from agentic_search.backends.base import BackendError


class NativeQueryRejected(BackendError):
    """The native query is not a single read-only statement."""


def guard_sql(query: str, dialect: str, max_rows: int) -> str:
    """Allow exactly one SELECT/set-operation with no DML/DDL/locking; cap or add LIMIT."""
    import sqlglot
    from sqlglot import exp

    try:
        statements = [s for s in sqlglot.parse(query, read=dialect) if s is not None]
    except sqlglot.errors.ParseError as exc:
        raise NativeQueryRejected(f"could not parse SQL: {str(exc)[:200]}") from exc
    if len(statements) != 1:
        raise NativeQueryRejected("native SQL must be exactly one statement")
    stmt = statements[0]
    if not isinstance(stmt, exp.Query):
        raise NativeQueryRejected(f"only SELECT queries are allowed, got {stmt.key.upper()}")
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter,
                 exp.Command, exp.Into, exp.Lock, exp.TruncateTable)
    for node in stmt.walk():
        if isinstance(node, forbidden):
            raise NativeQueryRejected(f"{type(node).__name__.upper()} is not allowed in native SQL")
    limit = stmt.args.get("limit")
    current = None
    if limit is not None:
        literal = limit.expression if hasattr(limit, "expression") else None
        if isinstance(literal, exp.Literal) and literal.is_int:
            current = int(literal.this)
    if current is None or current > max_rows:
        stmt = stmt.limit(max_rows)
    return stmt.sql(dialect=dialect)


_CYPHER_WRITE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|LOAD\s+CSV)\b", re.IGNORECASE)
_CYPHER_CALL = re.compile(r"\bCALL\s+([A-Za-z0-9_.]+)", re.IGNORECASE)
_CYPHER_ALLOWED_PROCS = {"db.index.fulltext.querynodes", "db.index.vector.querynodes",
                         "db.labels", "db.relationshiptypes", "db.propertykeys"}
_CYPHER_LIMIT = re.compile(r"\bLIMIT\s+(\d+)\s*;?\s*$", re.IGNORECASE)
_CYPHER_STRING = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")


def guard_cypher(query: str, max_rows: int) -> str:
    """Reject write clauses and non-allowlisted procedures; cap or add a trailing LIMIT."""
    code = _CYPHER_STRING.sub("''", query)  # ignore keywords inside string literals
    if ";" in code.strip().rstrip(";"):
        raise NativeQueryRejected("native Cypher must be exactly one statement")
    match = _CYPHER_WRITE.search(code)
    if match:
        raise NativeQueryRejected(f"{match.group(1).upper()} is not allowed in native Cypher")
    for proc in _CYPHER_CALL.findall(code):
        if proc.lower() not in _CYPHER_ALLOWED_PROCS:
            raise NativeQueryRejected(f"procedure {proc} is not allowed in native Cypher")
    q = query.strip().rstrip(";").rstrip()
    limit = _CYPHER_LIMIT.search(q)
    if limit is None:
        return f"{q} LIMIT {max_rows}"
    if int(limit.group(1)) > max_rows:
        return q[: limit.start()] + f"LIMIT {max_rows}"
    return q


_DSL_TOP_KEYS = {"query", "size", "from", "_source", "sort", "aggs", "aggregations",
                 "highlight", "track_total_hits"}


def _walk_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [k for k in value] + [k2 for v in value.values() for k2 in _walk_keys(v)]
    if isinstance(value, list):
        return [k for v in value for k in _walk_keys(v)]
    return []


def guard_opensearch(body_json: str, max_rows: int) -> dict[str, Any]:
    """Accept only a search body with known top-level keys, no scripts; cap size."""
    try:
        body = json.loads(body_json)
    except json.JSONDecodeError as exc:
        raise NativeQueryRejected(f"native OpenSearch query must be a JSON search body: {exc}") from exc
    if not isinstance(body, dict):
        raise NativeQueryRejected("native OpenSearch query must be a JSON object")
    unknown = set(body) - _DSL_TOP_KEYS
    if unknown:
        raise NativeQueryRejected(f"top-level keys not allowed: {sorted(unknown)}")
    scripted = [k for k in _walk_keys(body) if "script" in k.lower()]
    if scripted:
        raise NativeQueryRejected(f"scripts are not allowed ({scripted[0]})")
    size = body.get("size")
    body["size"] = max_rows if not isinstance(size, int) or size > max_rows else size
    return body
