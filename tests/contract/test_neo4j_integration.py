"""Neo4j-specific integration checks beyond the shared contract."""

import os

import pytest

from agentic_search.backends.base import BackendError
from agentic_search.core.types import Fetch

pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(os.environ.get("AGENTIC_SEARCH_INTEGRATION") != "1",
                                 reason="set AGENTIC_SEARCH_INTEGRATION=1 with docker compose up")]


@pytest.fixture
async def neo():
    from .seed_neo4j import make_neo4j
    b = await make_neo4j()
    yield b
    await b.close()


async def test_server_refuses_writes_in_read_sessions(neo):
    await neo.discover()
    with pytest.raises(BackendError, match="AccessMode|read access"):
        await neo._read("MATCH (n:docs {id: 'd1'}) SET n.title = 'x'")
    [still] = await neo.execute(Fetch(source="neo", collection="docs", doc_ids=["d1"]))
    assert still.metadata["id"] == "d1"


async def test_namespaced_fetch_routes_by_prefix(neo):
    [hit] = await neo.execute(Fetch(source="neo", doc_ids=["docs/d3"]))
    assert hit.doc_id == "docs/d3"
    with pytest.raises(BackendError, match="collection required"):
        await neo.execute(Fetch(source="neo", doc_ids=["d3"]))
