"""Unit tests for PostgresBackend (no docker required)."""

import sys

import pytest

from agentic_search.backends.base import BackendError
from agentic_search.backends.postgres import PostgresBackend, _fresh_password_class
from agentic_search.core.secrets import scrub


@pytest.mark.asyncio
async def test_postgres_connection_error_surfaces_as_backend_error():
    """Verify that unreachable Postgres raises BackendError, not driver-specific errors."""
    # Use port 1 (nothing listening) with short timeout for fast failure
    backend = PostgresBackend(
        "pg",
        "postgresql://user:secretpw9@127.0.0.1:1/testdb",
        connect_timeout_s=2
    )

    # Attempt to discover should raise BackendError, not psycopg/psycopg_pool errors
    with pytest.raises(BackendError) as excinfo:
        await backend.discover()

    # Verify that the error message does not contain the password after scrubbing
    error_str = str(excinfo.value)
    scrubbed = scrub(error_str)
    assert "secretpw9" not in scrubbed, (
        f"Password leaked in scrubbed error message: {scrubbed}"
    )

    await backend.close()


@pytest.mark.asyncio
async def test_postgres_missing_extra_raises_backend_error(monkeypatch):
    """Verify that missing postgres extra raises BackendError, not ImportError."""
    # Simulate missing psycopg by making import fail
    monkeypatch.setitem(sys.modules, "psycopg", None)

    backend = PostgresBackend("pg", "postgresql://u:p@h/db")

    # Attempt to discover should raise BackendError mentioning postgres extra
    with pytest.raises(BackendError) as excinfo:
        await backend.discover()

    error_str = str(excinfo.value)
    assert "postgres" in error_str and "extra" in error_str, (
        f"Error message should mention postgres extra, got: {error_str}"
    )

    await backend.close()


def test_percent_encoded_password_is_registered():
    PostgresBackend("pg", "postgresql://u:p%40ssw0rd-pg9@h/db")
    assert "p@ssw0rd-pg9" not in scrub("auth failed: p@ssw0rd-pg9")


@pytest.mark.asyncio
async def test_pool_open_cancelled_closes_pool(monkeypatch):
    import asyncio

    import agentic_search.backends.postgres as pg

    closed = []

    class FakePsycopg:
        class AsyncConnection:
            pass

    class FakePool:
        def __init__(self, *args, **kwargs):
            pass

        async def open(self, wait, timeout):
            raise asyncio.CancelledError()

        async def close(self):
            closed.append(True)

    monkeypatch.setattr(pg, "_require_psycopg", lambda: (FakePsycopg(), None, FakePool))
    backend = PostgresBackend("pg", "postgresql://u:p@h/db")
    with pytest.raises(asyncio.CancelledError):
        await backend._get_pool()
    assert closed == [True] and backend._pool is None


def test_lexical_uses_stored_tsvector_column():
    from agentic_search.backends.sql_backend import TableInfo
    from agentic_search.core.types import FieldSpec, FieldType, Lexical

    b = PostgresBackend("pg", "postgresql://u:p@h/db")
    table = TableInfo(name="docs", id_column="id", fields=[
        FieldSpec(name="id", type=FieldType.KEYWORD), FieldSpec(name="body", type=FieldType.TEXT, searchable=True)])
    b._tsv = {"docs": "search"}
    sql, _ = b._lexical_sql(Lexical(source="pg", text="headache"), table, 5)
    assert '"search" @@ websearch_to_tsquery' in sql and "to_tsvector" not in sql
    sql2, _ = b._lexical_sql(Lexical(source="pg", text="headache", fields=["body"]), table, 5)
    assert "to_tsvector('english', concat_ws(' ', \"body\"))" in sql2


@pytest.mark.asyncio
async def test_password_provider_is_asked_at_each_connect(monkeypatch):
    psycopg = pytest.importorskip("psycopg")
    seen = []

    async def fake_connect(cls, conninfo="", **kwargs):
        seen.append(kwargs["password"])
        return object()

    monkeypatch.setattr(psycopg.AsyncConnection, "connect", classmethod(fake_connect))
    tokens = iter(["tok-first-1234", "tok-second-5678"])

    async def provider():
        return next(tokens)

    connection_class = _fresh_password_class(psycopg, provider)
    await connection_class.connect("postgresql://sa@127.0.0.1/db")
    await connection_class.connect("postgresql://sa@127.0.0.1/db")
    assert seen == ["tok-first-1234", "tok-second-5678"]
    assert "tok-second-5678" not in scrub("leaked tok-second-5678")


@pytest.mark.asyncio
async def test_backend_uses_the_provider_for_its_pool(monkeypatch):
    psycopg = pytest.importorskip("psycopg")
    asked = []

    async def fake_connect(cls, conninfo="", **kwargs):
        raise psycopg.OperationalError("refused")

    monkeypatch.setattr(psycopg.AsyncConnection, "connect", classmethod(fake_connect))

    async def provider():
        asked.append(True)
        return "tok-pool-9012"

    backend = PostgresBackend("pg", "postgresql://sa@127.0.0.1:1/db", password=provider,
                              connect_timeout_s=1)
    with pytest.raises(BackendError):
        await backend.discover()
    assert asked
    await backend.close()
