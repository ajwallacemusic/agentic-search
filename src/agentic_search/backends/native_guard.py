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


_SQL_FUNC_DENYLIST = re.compile(
    r"^(pg_(sleep|advisory|try_advisory|read|ls|stat_file|terminate|cancel|reload|rotate|switch|promote|create|drop|logical)|"
    r"lo_|set_config|dblink|nextval|setval|get_lock|release_lock|is_used_lock|sleep|benchmark|load_file|sys_exec|sys_eval|xp_|query_to_xml)",
    re.IGNORECASE
)


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
        # Check for side-effecting functions
        if isinstance(node, exp.Func):
            func_name = node.name if isinstance(node, exp.Anonymous) else node.sql_name()
            if _SQL_FUNC_DENYLIST.match(func_name):
                raise NativeQueryRejected(f"function {func_name} is not allowed in native SQL")
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


def _cypher_code(query: str) -> str:
    """Scan query left-to-right, reject comments, return sanitized code with string/backtick placeholders."""
    result = []
    i = 0
    while i < len(query):
        ch = query[i]

        # Handle single-quoted strings
        if ch == "'":
            result.append("''")
            i += 1
            while i < len(query):
                if query[i] == "\\":
                    i += 2  # Skip escaped char
                elif query[i] == "'":
                    i += 1
                    break
                else:
                    i += 1
            if i > len(query):
                raise NativeQueryRejected("unterminated single-quoted string in native Cypher")
            continue

        # Handle double-quoted strings
        if ch == '"':
            result.append('""')
            i += 1
            while i < len(query):
                if query[i] == "\\":
                    i += 2  # Skip escaped char
                elif query[i] == '"':
                    i += 1
                    break
                else:
                    i += 1
            if i > len(query):
                raise NativeQueryRejected("unterminated double-quoted string in native Cypher")
            continue

        # Handle backtick identifiers
        if ch == "`":
            result.append("`x`")
            i += 1
            while i < len(query):
                if query[i] == "\\":
                    i += 2  # Skip escaped char
                elif query[i] == "`":
                    i += 1
                    break
                else:
                    i += 1
            if i > len(query):
                raise NativeQueryRejected("unterminated backtick identifier in native Cypher")
            continue

        # Handle line comments
        if i + 1 < len(query) and query[i:i+2] == "//":
            raise NativeQueryRejected("comments are not allowed in native Cypher")

        # Handle block comments
        if i + 1 < len(query) and query[i:i+2] == "/*":
            raise NativeQueryRejected("comments are not allowed in native Cypher")

        result.append(ch)
        i += 1

    return "".join(result)


_CYPHER_LIMIT = re.compile(r"\bLIMIT\s+(\d+)\s*;?\s*$", re.IGNORECASE)


def guard_cypher(query: str, max_rows: int) -> str:
    """Reject write clauses and non-allowlisted procedures; wrap and cap with LIMIT."""
    code = _cypher_code(query)
    if ";" in code.strip().rstrip(";"):
        raise NativeQueryRejected("native Cypher must be exactly one statement")
    match = _CYPHER_WRITE.search(code)
    if match:
        raise NativeQueryRejected(f"{match.group(1).upper()} is not allowed in native Cypher")
    for proc in _CYPHER_CALL.findall(code):
        if proc.lower() not in _CYPHER_ALLOWED_PROCS:
            raise NativeQueryRejected(f"procedure {proc} is not allowed in native Cypher")
    q = query.strip().rstrip(";").rstrip()
    # Extract existing LIMIT to use minimum of original and max_rows
    limit_match = _CYPHER_LIMIT.search(q)
    limit_value = max_rows
    if limit_match:
        original_limit = int(limit_match.group(1))
        limit_value = min(original_limit, max_rows)
        # Remove the original LIMIT from the query as we'll wrap it
        q = q[:limit_match.start()].rstrip()
    return f"CALL {{ {q} }} RETURN * LIMIT {limit_value}"


_DSL_TOP_KEYS = {"query", "size", "from", "_source", "sort", "aggs", "aggregations",
                 "highlight", "track_total_hits"}


def _walk_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [k for k in value] + [k2 for v in value.values() for k2 in _walk_keys(v)]
    if isinstance(value, list):
        return [k for v in value for k in _walk_keys(v)]
    return []


def guard_opensearch(body_json: str, max_rows: int) -> dict[str, Any]:
    """Accept only a search body with known top-level keys, no scripts; cap size and from."""
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

    # Validate and cap size: must be int (not bool), >= 0, and <= max_rows
    size = body.get("size")
    if isinstance(size, int) and not isinstance(size, bool) and size >= 0 and size <= max_rows:
        body["size"] = size
    else:
        body["size"] = max_rows

    # Validate from parameter
    from_val = body.get("from")
    if from_val is not None:
        if not isinstance(from_val, int) or from_val < 0:
            raise NativeQueryRejected("from must be a non-negative integer")
        # Cap from + size at 10000 to prevent deep pagination DoS
        if from_val + body["size"] > 10000:
            raise NativeQueryRejected(f"from + size exceeds maximum (10000)")

    return body
