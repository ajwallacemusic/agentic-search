import httpx
import pytest

from agentic_search.core.secrets import scrub
from agentic_search.core.types import ImagePart, TextPart
from agentic_search.embedders.auth import Bearer
from agentic_search.embedders.base import EmbedderError
from agentic_search.embedders.openai_compat import OpenAICompatEmbedder
from agentic_search.embedders.tei import TEIEmbedder

from .mock_http import mock


async def test_retries_transient_then_succeeds():
    calls = {"n": 0}

    def respond(req):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="overloaded")
        return httpx.Response(200, json=[[1.0]])

    e = TEIEmbedder("http://tei", 1, client=mock(respond), backoff_s=0.001)
    assert await e.embed([TextPart(text="x")], "query") == [[1.0]]
    assert calls["n"] == 3


async def test_errors_are_embedder_errors_and_scrubbed():
    e = TEIEmbedder("http://tei", 1, client=mock(lambda r: httpx.Response(401, text="bad key sk-live-secret99")),
                    auth=Bearer("sk-live-secret99"), backoff_s=0.001)
    with pytest.raises(EmbedderError) as exc:
        await e.embed([TextPart(text="x")], "query")
    assert "401" in str(exc.value) and "sk-live-secret99" not in str(exc.value)
    wrong_dim = TEIEmbedder("http://tei", 3, client=mock(lambda r: httpx.Response(200, json=[[1.0]])))
    with pytest.raises(EmbedderError, match="dim"):
        await wrong_dim.embed([TextPart(text="x")], "query")
    text_only = TEIEmbedder("http://tei", 1, client=mock(lambda r: httpx.Response(200, json=[[1.0]])))
    with pytest.raises(EmbedderError, match="image"):
        await text_only.embed([ImagePart(data=b"x")], "query")
    shape = OpenAICompatEmbedder("m", 1, client=mock(lambda r: httpx.Response(200, json={"nope": 1})))
    with pytest.raises(EmbedderError, match="shape"):
        await shape.embed([TextPart(text="x")], "query")
    assert scrub("sk-live-secret99") == "***"


async def test_transport_errors_retry_then_fail():
    def respond(req):
        raise httpx.ConnectError("refused")

    e = TEIEmbedder("http://tei", 1, client=mock(respond), max_retries=2, backoff_s=0.001)
    with pytest.raises(EmbedderError, match="ConnectError"):
        await e.embed([TextPart(text="x")], "query")
