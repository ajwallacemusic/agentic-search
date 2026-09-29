"""Unit tests for MySQLBackend (no docker required)."""

import sys

import pytest

from agentic_search.backends.base import BackendError
from agentic_search.backends.mysql import MySQLBackend
from agentic_search.core.secrets import scrub


def test_dsn_parsing_and_secret_registration():
    b = MySQLBackend("my", "mysql://reader:p%40ss-word@db.internal:3307/shop")
    assert b._conn_args == {"host": "db.internal", "port": 3307, "user": "reader",
                            "password": "p@ss-word", "db": "shop"}
    assert "p@ss-word" not in scrub("error for p@ss-word")


@pytest.mark.parametrize("dsn", ["postgresql://u:p@h/db", "mysql://u:p@h"])
def test_rejects_bad_dsn(dsn):
    with pytest.raises(ValueError):
        MySQLBackend("my", dsn)


@pytest.mark.asyncio
async def test_mysql_connection_error_surfaces_as_backend_error():
    """Verify that unreachable MySQL raises BackendError, not driver-specific errors."""
    # Use port 1 (nothing listening) with short timeout for fast failure
    backend = MySQLBackend(
        "my",
        "mysql://user:secretpw7@127.0.0.1:1/testdb",
        connect_timeout_s=2
    )

    # Attempt to discover should raise BackendError, not aiomysql/pymysql errors
    with pytest.raises(BackendError) as excinfo:
        await backend.discover()

    # Verify that the error message does not contain the password after scrubbing
    error_str = str(excinfo.value)
    scrubbed = scrub(error_str)
    assert "secretpw7" not in scrubbed, (
        f"Password leaked in scrubbed error message: {scrubbed}"
    )

    await backend.close()


@pytest.mark.asyncio
async def test_mysql_missing_extra_raises_backend_error(monkeypatch):
    """Verify that missing mysql extra raises BackendError, not ImportError."""
    # Simulate missing aiomysql by making import fail
    monkeypatch.setitem(sys.modules, "aiomysql", None)

    backend = MySQLBackend("my", "mysql://u:p@h/db")

    # Attempt to discover should raise BackendError mentioning mysql extra
    with pytest.raises(BackendError) as excinfo:
        await backend.discover()

    error_str = str(excinfo.value)
    assert "mysql" in error_str and "extra" in error_str, (
        f"Error message should mention mysql extra, got: {error_str}"
    )

    await backend.close()


@pytest.mark.asyncio
async def test_sessions_are_read_only_with_server_side_timeout(monkeypatch):
    import aiomysql

    seen = {}

    async def fake_create_pool(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(aiomysql, "create_pool", fake_create_pool)
    backend = MySQLBackend("my", "mysql://u:p@h/db", statement_timeout_ms="5000")
    assert backend.statement_timeout_ms == 5000
    await backend._get_pool()
    assert seen["init_command"] == "SET SESSION transaction_read_only=ON, max_execution_time=5000"
    assert MySQLBackend("my", "mysql://u:p@h/db").statement_timeout_ms == 30_000
