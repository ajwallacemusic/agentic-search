"""Read-only gate for `native_query`: SQL (via sqlglot), Cypher (keyword scan), OpenSearch DSL.

Every guard either returns a query that is safe to run (with a row cap applied) or raises
NativeQueryRejected. Adapters still connect with read-only sessions; this is the first line."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agentic_search.backends.base import BackendError


class NativeQueryRejected(BackendError):
    """The native query is not a single read-only statement."""


_SQL_FUNC_DENYLIST = re.compile(
    r"^(pg_(sleep|advisory|try_advisory|read|ls|stat_file|terminate|cancel|reload|rotate|switch|promote|create|drop|logical|notify)|"
    r"lo_|set_config|dblink|nextval|setval|get_lock|release_lock|is_used_lock|sleep|benchmark|load_file|sys_exec|sys_eval|xp_|query_to_xml)",
    re.IGNORECASE
)


DEFAULT_SQL_FUNCTIONS = frozenset({
    "COUNT", "COUNT_IF", "SUM", "AVG", "MIN", "MAX", "APPROX_DISTINCT",
    "LOWER", "UPPER", "LENGTH", "TRIM", "SUBSTRING", "CONCAT", "COALESCE",
    "STARTS_WITH", "ENDS_WITH", "CONTAINS", "REGEXP_LIKE",
    "CAST", "TRY_CAST", "IF", "CASE", "ROUND", "ABS", "SAFE_DIVIDE",
    "DATE", "EXTRACT", "DATE_TRUNC", "TIMESTAMP_TRUNC", "DATEDIFF", "CURRENT_DATE",
    "CURRENT_TIMESTAMP",
})
"""Uppercase sqlglot names (`Expression.sql_name()`, or the name of an unknown function).
BigQuery's CONTAINS_SUBSTR parses as CONTAINS and REGEXP_CONTAINS as REGEXP_LIKE."""


@dataclass(frozen=True)
class SqlAllowList:
    """What native SQL may read: tables and their columns, and functions by sqlglot name.

    `db` is the Postgres schema or BigQuery dataset every allowed table sits in; `catalog` is
    the BigQuery project. An unqualified table is read as one of these."""

    tables: Mapping[str, frozenset[str]]
    db: str | None = None
    catalog: str | None = None
    functions: frozenset[str] = DEFAULT_SQL_FUNCTIONS

    def __post_init__(self) -> None:
        if self.catalog is not None and self.db is None:
            raise ValueError("SqlAllowList: a catalog needs a db")

    def schema(self) -> dict[str, Any]:
        """The nested schema `qualify` resolves against: allowed columns only, sorted so `*`
        expands in a stable order."""
        schema: dict[str, Any] = {table: {column: "UNKNOWN" for column in sorted(columns)}
                                  for table, columns in self.tables.items()}
        if self.db is not None:
            schema = {self.db: schema}
        if self.catalog is not None:
            schema = {self.catalog: schema}
        return schema


def _apply_allow_list(stmt: Any, dialect: str, allow: SqlAllowList) -> Any:
    """Resolve every column against the allowed columns and return the resolved query.

    The resolved query is what runs, so `*` reads only allowed columns and a reference that
    `qualify` binds to a select alias never reaches a hidden column of the same name."""
    from sqlglot import exp
    from sqlglot.errors import OptimizeError
    from sqlglot.optimizer.qualify import qualify

    try:
        stmt = qualify(stmt, schema=allow.schema(), dialect=dialect, catalog=allow.catalog,
                       db=allow.db, validate_qualify_columns=True, quote_identifiers=False)
    except OptimizeError as exc:
        raise NativeQueryRejected(
            f"native SQL may use only the allowed tables and columns: {exc}") from exc
    # After a clean resolve, a column with no table is an alias reference. `qualify` leaves one
    # where it cannot tell alias from table column (HAVING) and where a bare table alias names
    # the whole row (`SELECT v FROM t v`); both can read a hidden column. Only ORDER BY on a
    # select alias, which `qualify` has already checked, may stay unqualified.
    for whole_row in stmt.find_all(exp.TableColumn):
        raise NativeQueryRejected(
            f"native SQL may use only the allowed tables and columns: "
            f"{whole_row.sql(dialect=dialect)} names a whole row, not a column")
    for column in stmt.find_all(exp.Column):
        if column.table:
            continue
        order = column.find_ancestor(exp.Order)
        if order is None or not isinstance(order.parent, exp.Query):
            raise NativeQueryRejected(
                f"native SQL may use only the allowed tables and columns: "
                f"{column.sql(dialect=dialect)} is not a qualified allowed column")
    ctes = {cte.alias_or_name for cte in stmt.find_all(exp.CTE)}
    for table in stmt.find_all(exp.Table):
        if table.name in ctes and not table.db:
            continue
        # A query naming no column, such as COUNT(*), resolves against any table; check each one.
        if (table.name not in allow.tables or (table.db or None) != allow.db
                or (table.catalog or None) != allow.catalog):
            raise NativeQueryRejected(f"table {table.sql(dialect=dialect)} is not allowed in native SQL")
    for func in stmt.find_all(exp.Func):
        if isinstance(func, exp.Connector):  # AND, OR and XOR are operators, not functions
            continue
        name = (func.name if isinstance(func, exp.Anonymous) else func.sql_name()).upper()
        if name not in allow.functions:
            raise NativeQueryRejected(f"function {name} is not allowed in native SQL")
    return stmt


def guard_sql(query: str, dialect: str, max_rows: int, allow: SqlAllowList | None = None) -> str:
    """Allow exactly one SELECT/set-operation with no DML/DDL/locking; cap or add LIMIT.
    With `allow`, also resolve it against the allowed tables, columns and functions."""
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
    if allow is not None:
        stmt = _apply_allow_list(stmt, dialect, allow)
    limit = stmt.args.get("limit")
    current = None
    if limit is not None:
        literal = limit.expression if hasattr(limit, "expression") else None
        if isinstance(literal, exp.Literal) and literal.is_int:
            current = int(literal.this)
    if current is None or current > max_rows:
        stmt = stmt.limit(max_rows)
    return stmt.sql(dialect=dialect, comments=False)


_CYPHER_WRITE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|LOAD\s+CSV)\b", re.IGNORECASE)
_CYPHER_CALL = re.compile(r"(?<!\.)\bCALL\b\s*", re.IGNORECASE)
_CYPHER_PROC_NAME = re.compile(r"[A-Za-z0-9_.]+")
_CYPHER_ALLOWED_PROCS = {"db.index.fulltext.querynodes", "db.index.vector.querynodes",
                         "db.labels", "db.relationshiptypes", "db.propertykeys"}


def _cypher_code(query: str) -> str:
    """Scan query left-to-right, reject comments, return sanitized code with string/backtick placeholders.

    Tracks: single/double-quoted strings (with backslash escapes), backtick identifiers (doubled backticks).
    Rejects comments (// and /*) and unterminated literals.
    """
    result = []
    i = 0
    in_string = None  # None, "'", '"', or "`"

    while i < len(query):
        ch = query[i]

        # Inside single-quoted string: handle backslash escapes and closing quote
        if in_string == "'":
            if ch == "\\":
                i += 2  # Skip escaped char
            elif ch == "'":
                result.append("''")
                in_string = None
                i += 1
            else:
                i += 1
            continue

        # Inside double-quoted string: handle backslash escapes and closing quote
        if in_string == '"':
            if ch == "\\":
                i += 2  # Skip escaped char
            elif ch == '"':
                result.append('""')
                in_string = None
                i += 1
            else:
                i += 1
            continue

        # Inside backtick identifier: Cypher uses doubled backticks, NOT backslash escapes
        if in_string == "`":
            if ch == "`":
                # Check if next char is also a backtick (escape sequence)
                if i + 1 < len(query) and query[i + 1] == "`":
                    i += 2  # Consume both backticks, stay inside
                else:
                    result.append("`x`")
                    in_string = None
                    i += 1
            else:
                i += 1
            continue

        # Outside strings: detect string/backtick starts
        if ch == "'":
            result.append("''")
            in_string = "'"
            i += 1
            continue

        if ch == '"':
            result.append('""')
            in_string = '"'
            i += 1
            continue

        if ch == "`":
            result.append("`x`")
            in_string = "`"
            i += 1
            continue

        # Outside strings: reject comments
        if i + 1 < len(query) and query[i:i+2] == "//":
            raise NativeQueryRejected("comments are not allowed in native Cypher")

        if i + 1 < len(query) and query[i:i+2] == "/*":
            raise NativeQueryRejected("comments are not allowed in native Cypher")

        result.append(ch)
        i += 1

    # Check for unterminated string or backtick at end of input
    if in_string is not None:
        raise NativeQueryRejected("unterminated string or identifier in native Cypher")

    return "".join(result)


_CYPHER_LIMIT = re.compile(r"\bLIMIT\s+(\d+)\s*;?\s*$", re.IGNORECASE)
_CYPHER_RETURN = re.compile(r"\bRETURN\b", re.IGNORECASE)


def _check_calls(code: str) -> None:
    """After CALL allow only a `{` subquery or a plain, allowlisted dotted procedure name.
    Anything else (a backtick-quoted name, `CALL (x) {`, ...) could hide a procedure."""
    for call in _CYPHER_CALL.finditer(code):
        rest = code[call.end():]
        if rest.startswith("{"):
            continue
        name = _CYPHER_PROC_NAME.match(rest)
        if name is None or rest[name.end():name.end() + 1] == "`":
            raise NativeQueryRejected(
                "CALL must be followed by a { subquery } or a plain allowlisted procedure name")
        if name.group(0).lower() not in _CYPHER_ALLOWED_PROCS:
            raise NativeQueryRejected(f"procedure {name.group(0)} is not allowed in native Cypher")


def _check_braces(code: str) -> None:
    """Unbalanced braces could close the `CALL { ... }` wrapper early and escape its LIMIT."""
    depth = 0
    for ch in code:
        depth += {"{": 1, "}": -1}.get(ch, 0)
        if depth < 0:
            break
    if depth != 0:
        raise NativeQueryRejected("unbalanced braces in native Cypher")


def guard_cypher(query: str, max_rows: int) -> str:
    """Reject write clauses and non-allowlisted procedures; wrap and cap with LIMIT."""
    # Neo4j decodes \uXXXX escapes before tokenising, anywhere in the query, so an escape could
    # hide a keyword, a brace or a quote from every check below.
    if re.search(r"\\u", query, re.IGNORECASE):
        raise NativeQueryRejected("unicode escapes are not allowed in native Cypher")
    code = _cypher_code(query)
    if ";" in code.strip().rstrip(";"):
        raise NativeQueryRejected("native Cypher must be exactly one statement")
    match = _CYPHER_WRITE.search(code)
    if match:
        raise NativeQueryRejected(f"{match.group(1).upper()} is not allowed in native Cypher")
    _check_calls(code)
    _check_braces(code)
    q = query.strip().rstrip(";").rstrip()
    # Extract existing LIMIT to use minimum of original and max_rows
    limit_match = _CYPHER_LIMIT.search(q)
    limit_value = max_rows
    if limit_match:
        original_limit = int(limit_match.group(1))
        limit_value = min(original_limit, max_rows)
        # Remove the original LIMIT from the query as we'll wrap it
        q = q[:limit_match.start()].rstrip()
    # If the query doesn't have RETURN (e.g., procedure-only CALL), add it
    if not _CYPHER_RETURN.search(q):
        q = f"{q} RETURN *"
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
    # terms lookup, more_like_this, percolate etc. can read documents from other indices
    if any(k in ("index", "_index") for k in _walk_keys(body)):
        raise NativeQueryRejected("references to other indices (index/_index) are not allowed")

    # Validate and cap size: must be int (not bool), >= 0, and <= max_rows
    size = body.get("size")
    if isinstance(size, int) and not isinstance(size, bool) and size >= 0 and size <= max_rows:
        body["size"] = size
    else:
        body["size"] = max_rows

    # Validate from parameter: must be int (not bool), and non-negative
    from_val = body.get("from")
    if from_val is not None:
        if not isinstance(from_val, int) or isinstance(from_val, bool) or from_val < 0:
            raise NativeQueryRejected("from must be a non-negative integer")
        # Cap from + size at 10000 to prevent deep pagination DoS
        if from_val + body["size"] > 10000:
            raise NativeQueryRejected("from + size exceeds maximum (10000)")

    return body
