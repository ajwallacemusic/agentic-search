"""Keep credentials out of error messages and traces.

Values are registered when config resolves `*_env` keys and when adapters receive DSNs or
passwords; `scrub()` replaces them (and any URL userinfo) before text reaches a ToolError,
the trace, or `Harness.setup_errors`."""

from __future__ import annotations

import re
from typing import Any

MASK = "***"
_MIN_SECRET_LEN = 4
_secrets: set[str] = set()
_URL_USERINFO = re.compile(r"(?P<scheme>\b[a-zA-Z][a-zA-Z0-9+.-]*://)[^/@\s]+@")


def register_secret(value: str | None) -> None:
    """Remember a secret so scrub() masks it. Very short values are ignored (too many false hits)."""
    if value and len(value) >= _MIN_SECRET_LEN:
        _secrets.add(value)


def clear_secrets() -> None:
    """For tests."""
    _secrets.clear()


def scrub(text: str) -> str:
    for secret in sorted(_secrets, key=len, reverse=True):
        text = text.replace(secret, MASK)
    return _URL_USERINFO.sub(lambda m: f"{m.group('scheme')}{MASK}@", text)


def scrub_data(value: Any) -> Any:
    """Recursively scrub every string inside dicts, lists and tuples."""
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, dict):
        return {k: scrub_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(scrub_data(v) for v in value)
    return value
