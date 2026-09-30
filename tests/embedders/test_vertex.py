import base64
import json

import httpx

from agentic_search.core.types import ImagePart, TextPart
from agentic_search.embedders.auth import Bearer
from agentic_search.embedders.vertex import VertexEmbedder

from .mock_http import Recorder, mock


async def test_vertex_text_request():
    rec = Recorder(lambda req: httpx.Response(200, json={"predictions": [
        {"embeddings": {"values": [0.5, 0.5]}}]}))
    e = VertexEmbedder("gemini-embedding-001", 2, project="proj-1", location="us-east1",
                       auth=Bearer("ya29.token-abc"), client=mock(rec))
    assert e.batch_size == 1 and e.id == "vertex:gemini-embedding-001"
    await e.embed([TextPart(text="headache")], "query")
    assert str(rec.requests[0].url) == (
        "https://us-east1-aiplatform.googleapis.com/v1/projects/proj-1/locations/us-east1"
        "/publishers/google/models/gemini-embedding-001:predict")
    assert rec.body() == {"instances": [{"content": "headache", "task_type": "RETRIEVAL_QUERY"}],
                          "parameters": {"outputDimensionality": 2, "autoTruncate": True}}


async def test_vertex_multimodal_image_and_text():
    def respond(req):
        inst = json.loads(req.content)["instances"][0]
        key = "imageEmbedding" if "image" in inst else "textEmbedding"
        return httpx.Response(200, json={"predictions": [{key: [1.0, 0.0] if key == "imageEmbedding" else [0.0, 1.0]}]})

    rec = Recorder(respond)
    e = VertexEmbedder("multimodalembedding@001", 2, project="p", auth=Bearer("tok-12345"),
                       client=mock(rec))
    vecs = await e.embed([ImagePart(data=b"\x89PNG"), TextPart(text="x-ray")], "query")
    assert vecs == [[1.0, 0.0], [0.0, 1.0]]
    assert rec.body(0) == {"instances": [{"image": {"bytesBase64Encoded": base64.b64encode(b"\x89PNG").decode()}}],
                           "parameters": {"dimension": 2}}
