"""Filter AST utilities. Plans 2 and 3 add translators to backend dialects here."""

from __future__ import annotations

import operator
from typing import Any, Callable

from agentic_search.core.types import And, Contains, Eq, Exists, Filter, In, Not, Or, Range

_MISSING = object()


def _lookup(record: dict[str, Any], field: str) -> Any:
    if field in record:
        return record[field]
    cur: Any = record
    for part in field.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return _MISSING
    return cur


def _compare(value: Any, bound: Any, fn: Callable[[Any, Any], bool]) -> bool:
    try:
        return bool(fn(value, bound))
    except TypeError:
        return False


def matches(f: Filter, record: dict[str, Any]) -> bool:
    """Evaluate a filter against a flat or nested dict (used by in-memory backends)."""
    if isinstance(f, And):
        return all(matches(c, record) for c in f.clauses)
    if isinstance(f, Or):
        return any(matches(c, record) for c in f.clauses)
    if isinstance(f, Not):
        return not matches(f.clause, record)
    value = _lookup(record, f.field)
    if isinstance(f, Exists):
        return value is not _MISSING and value is not None
    if value is _MISSING or value is None:
        return False
    if isinstance(f, Eq):
        return f.value in value if isinstance(value, list) else value == f.value
    if isinstance(f, In):
        if isinstance(value, list):
            return any(v in f.values for v in value)
        return value in f.values
    if isinstance(f, Contains):
        if isinstance(value, list):
            return f.value in value
        return f.value.lower() in str(value).lower()
    if isinstance(f, Range):
        checks = ((f.gte, operator.ge), (f.gt, operator.gt), (f.lte, operator.le), (f.lt, operator.lt))
        return all(_compare(value, bound, fn) for bound, fn in checks if bound is not None)
    raise TypeError(f"unknown filter node {type(f).__name__}")


def filter_fields(f: Filter) -> set[str]:
    """Every field name a filter references."""
    if isinstance(f, (And, Or)):
        return set().union(*(filter_fields(c) for c in f.clauses))
    if isinstance(f, Not):
        return filter_fields(f.clause)
    return {f.field}
