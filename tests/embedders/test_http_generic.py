import base64
import json

import httpx
import pytest

from agentic_search.core.types import ImagePart, TextPart
from agentic_search.embedders.auth import Bearer
from agentic_search.embedders.http_generic import GenericHttpEmbedder, json_path, render

from .mock_http import Recorder, mock


def test_json_path_and_render():
    doc = {"predictions": [{"embedding": [1, 2]}, {"embedding": [3, 4]}], "a": {"b-c": 5}}
    assert json_path(doc, "$.predictions[*].embedding") == [[1, 2], [3, 4]]
    assert json_path(doc, "$.predictions[1].embedding") == [[3, 4]]
    assert json_path(doc, "$.a['b-c']") == [5]
    assert json_path(doc, "$.missing") == []
    with pytest.raises(ValueError):
        json_path(doc, "predictions")
    assert render({"x": ["{{text}}!", 3]}, {"{{text}}": "hi"}) == {"x": ["hi!", 3]}


async def test_generic_http_medsiglip_style():
    def respond(req):
        inst = json.loads(req.content)["instances"][0]
        return httpx.Response(200, json={"predictions": [{"embedding": [0.1, 0.2] if "image_b64" in inst else [0.3, 0.4]}]})

    rec = Recorder(respond)
    e = GenericHttpEmbedder(
        "https://medsiglip.azureml.net/score", 2, id="azure:medsiglip-448", auth=Bearer("eyJ.azure-tok"),
        request={"image": {"instances": [{"image_b64": "{{b64}}", "mime": "{{mime}}"}]},
                 "text": {"instances": [{"text": "{{text}}"}]}},
        response_path="$.predictions[*].embedding", client=mock(rec))
    vecs = await e.embed([ImagePart(data=b"img", mime="image/jpeg"), TextPart(text="pneumonia")], "document")
    assert vecs == [[0.1, 0.2], [0.3, 0.4]]
    assert rec.body(0) == {"instances": [{"image_b64": base64.b64encode(b"img").decode(), "mime": "image/jpeg"}]}
    assert rec.body(1) == {"instances": [{"text": "pneumonia"}]}
    assert rec.requests[0].headers["authorization"] == "Bearer eyJ.azure-tok"


def test_render_is_single_pass():
    """Substituted values are never re-scanned: user text containing {{b64}} stays as typed."""
    values = {"{{text}}": "say {{b64}} and {{mime}}", "{{b64}}": "QUJD", "{{mime}}": "image/png"}
    assert render({"t": "{{text}}|{{b64}}"}, values) == {"t": "say {{b64}} and {{mime}}|QUJD"}
