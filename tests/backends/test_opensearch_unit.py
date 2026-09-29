"""Unit tests for OpenSearchBackend (no docker required)."""

import sys

import pytest

from agentic_search.backends.base import BackendError
from agentic_search.backends.opensearch import filter_dsl
from agentic_search.core.secrets import scrub
from agentic_search.core.types import And, Contains, Eq, Exists, In, Not, Or, Range


def test_filter_dsl():
    f = And(clauses=[Eq(field="type", value="drug"), In(field="year", values=[2020, 2021]),
                     Or(clauses=[Range(field="year", gte=2020), Not(clause=Exists(field="title"))]),
                     Contains(field="type", value="ru")])
    assert filter_dsl(f) == {"bool": {"filter": [
        {"term": {"type": "drug"}},
        {"terms": {"year": [2020, 2021]}},
        {"bool": {"should": [{"range": {"year": {"gte": 2020}}},
                             {"bool": {"must_not": [{"exists": {"field": "title"}}]}}],
                  "minimum_should_match": 1}},
        {"wildcard": {"type": {"value": "*ru*", "case_insensitive": True}}},
    ]}}
    assert filter_dsl(Range(field="year")) == {"match_all": {}}


@pytest.mark.asyncio
async def test_opensearch_connection_error_surfaces_as_backend_error():
    """Verify that unreachable OpenSearch raises BackendError, not driver-specific errors."""
    from agentic_search.backends.opensearch import OpenSearchBackend

    # Use port 1 (nothing listening) with invalid credentials for fast failure
    backend = OpenSearchBackend("os", "http://admin:secretpw5@127.0.0.1:1")

    try:
        # Attempt to discover should raise BackendError, not opensearchpy errors
        with pytest.raises(BackendError) as excinfo:
            await backend.discover()

        # Verify that the error message does not contain the password after scrubbing
        error_str = str(excinfo.value)
        scrubbed = scrub(error_str)
        assert "secretpw5" not in scrubbed, (
            f"Password leaked in scrubbed error message: {scrubbed}"
        )
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_opensearch_missing_extra_raises_backend_error(monkeypatch):
    """Verify that missing opensearch extra raises BackendError, not ImportError."""
    from agentic_search.backends.opensearch import OpenSearchBackend

    # Simulate missing opensearchpy by making import fail
    monkeypatch.setitem(sys.modules, "opensearchpy", None)

    backend = OpenSearchBackend("os", "http://admin:pw@127.0.0.1:9200")

    # Attempt to discover should raise BackendError mentioning opensearch extra
    with pytest.raises(BackendError) as excinfo:
        await backend.discover()

    error_str = str(excinfo.value)
    assert "opensearch" in error_str and "extra" in error_str, (
        f"Error message should mention opensearch extra, got: {error_str}"
    )

    await backend.close()
