"""Unit tests for PostgresBackend (no docker required)."""

import pytest

from agentic_search.backends.base import BackendError
from agentic_search.backends.postgres import PostgresBackend
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
