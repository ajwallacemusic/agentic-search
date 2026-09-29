"""Unit tests for OpenSearchBackend (no docker required)."""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_search.backends.base import BackendError
from agentic_search.backends.opensearch import (
    OpenSearchBackend,
    filter_dsl,
)
from agentic_search.core.secrets import scrub
from agentic_search.core.types import (
    Aggregate,
    And,
    Contains,
    Eq,
    Exists,
    In,
    Not,
    Or,
    Range,
)


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


def test_password_registration():
    """Verify that constructor registers the password for scrubbing."""
    # Create backend with a test password
    backend = OpenSearchBackend("os", "http://admin:secretpw5@127.0.0.1:9200")

    # Verify that the password is registered (scrubbing should hide it)
    scrubbed = scrub("leak secretpw5 here")
    assert scrubbed == "leak *** here", (
        f"Password should be scrubbed by registration, got: {scrubbed}"
    )


@pytest.mark.asyncio
async def test_opensearch_connection_error_surfaces_as_backend_error():
    """Verify that unreachable OpenSearch raises BackendError, not driver-specific errors."""
    # Use port 1 (nothing listening) with invalid credentials for fast failure
    backend = OpenSearchBackend("os", "http://admin:secretpw5@127.0.0.1:1")

    try:
        # Attempt to discover should raise BackendError, not opensearchpy errors
        with pytest.raises(BackendError) as excinfo:
            await backend.discover()

        # Verify that the error message is a BackendError type
        assert isinstance(excinfo.value, BackendError)
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_opensearch_missing_extra_raises_backend_error(monkeypatch):
    """Verify that missing opensearch extra raises BackendError, not ImportError."""
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


@pytest.mark.asyncio
async def test_injected_client_method_raises_value_error():
    """Verify that injected client raising ValueError becomes BackendError."""
    # Create a mock client that raises ValueError
    mock_client = MagicMock()
    mock_client.cat = MagicMock()
    mock_client.cat.indices = AsyncMock(side_effect=ValueError("bad url format"))
    mock_client.close = AsyncMock()

    backend = OpenSearchBackend("os", "http://admin:pw@127.0.0.1:9200", client=mock_client)

    try:
        with pytest.raises(BackendError) as excinfo:
            await backend.discover()

        assert "ValueError" in str(excinfo.value)
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_injected_client_method_raises_opensearch_exception():
    """Verify that injected client raising opensearchpy exception becomes BackendError."""
    # Create a fake exception that looks like opensearchpy exception
    class FakeOpenSearchException(Exception):
        __module__ = "opensearchpy.exceptions"

    mock_client = MagicMock()
    mock_client.cat = MagicMock()
    mock_client.cat.indices = AsyncMock(side_effect=FakeOpenSearchException("connection failed"))
    mock_client.close = AsyncMock()

    backend = OpenSearchBackend("os", "http://admin:pw@127.0.0.1:9200", client=mock_client)

    try:
        with pytest.raises(BackendError) as excinfo:
            await backend.discover()

        assert "FakeOpenSearchException" in str(excinfo.value)
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_empty_metrics_aggregate_raises_backend_error():
    """Verify that aggregation with no metrics raises BackendError."""
    # Create a mock client with minimal responses for discovery
    mock_client = MagicMock()
    mock_client.cat = MagicMock()
    mock_client.indices = MagicMock()
    mock_client.cat.indices = AsyncMock(return_value=[{"index": "test_index"}])
    mock_client.indices.get_mapping = AsyncMock(return_value={
        "test_index": {"mappings": {"properties": {"name": {"type": "keyword"}}}}
    })
    mock_client.count = AsyncMock(return_value={"count": 10})
    mock_client.search = AsyncMock(return_value={
        "hits": {"hits": []},
        "aggregations": {
            "s0": {"buckets": [{"key": "value1", "doc_count": 5}]}
        }
    })
    mock_client.close = AsyncMock()

    backend = OpenSearchBackend("os", "http://admin:pw@127.0.0.1:9200", client=mock_client)

    try:
        # Discover first to populate indices
        await backend.discover()

        # Now attempt an aggregate with empty metrics using model_construct
        # to bypass pydantic's default_factory for metrics
        op = Aggregate.model_construct(
            source="test",
            collection="test_index",
            group_by=["name"],
            metrics=[]
        )
        with pytest.raises(BackendError) as excinfo:
            await backend.execute(op)

        assert "at least one metric" in str(excinfo.value)
    finally:
        await backend.close()
