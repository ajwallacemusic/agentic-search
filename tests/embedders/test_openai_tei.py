import json

import httpx

from agentic_search.core.types import TextPart
from agentic_search.embedders.openai_compat import OpenAICompatEmbedder
from agentic_search.embedders.tei import TEIEmbedder

from .mock_http import Recorder, mock


async def test_openai_compat_request_and_order():
    def respond(req):
        n = len(json.loads(req.content)["input"])
        return httpx.Response(200, json={"data": [{"index": i, "embedding": [float(i), 1.0]}
                                                  for i in reversed(range(n))]})

    rec = Recorder(respond)
    e = OpenAICompatEmbedder("text-embedding-3-small", 2, base_url="https://llm.local/v1/",
                             api_key="sk-test-1234", dimensions=2, query_prefix="query: ",
                             batch_size=2, client=mock(rec))
    vecs = await e.embed([TextPart(text="a"), TextPart(text="b"), TextPart(text="c")], "query")
    assert vecs == [[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]]
    assert str(rec.requests[0].url) == "https://llm.local/v1/embeddings"
    assert rec.requests[0].headers["authorization"] == "Bearer sk-test-1234"
    assert rec.body(0) == {"model": "text-embedding-3-small", "input": ["query: a", "query: b"],
                           "dimensions": 2}
    assert e.id == "openai:text-embedding-3-small"


async def test_tei_request():
    rec = Recorder(lambda req: httpx.Response(200, json=[[0.1, 0.2, 0.3]]))
    e = TEIEmbedder("http://tei:8080", 3, id="tei:bge", document_prefix="passage: ", client=mock(rec))
    assert await e.embed([TextPart(text="x")], "document") == [[0.1, 0.2, 0.3]]
    assert str(rec.requests[0].url) == "http://tei:8080/embed"
    assert rec.body() == {"inputs": ["passage: x"], "normalize": True, "truncate": True}
