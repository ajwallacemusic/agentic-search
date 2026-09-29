import numpy as np
import pytest

from agentic_search.core.types import ImagePart, Modality, TextPart
from agentic_search.embedders.base import (
    CachedEmbedder, EmbedderRegistry, cosine_scores, normalize_rows, supports,
)
from agentic_search.embedders.local import HashEmbedder


async def test_hash_embedder_similarity_and_determinism():
    e = HashEmbedder(dim=128)
    a, b, c = await e.embed(
        [TextPart(text="cat sat on mat"), TextPart(text="the cat on a mat"),
         TextPart(text="quarterly revenue report")], "document")
    assert len(a) == 128
    assert np.dot(a, b) > np.dot(a, c)
    assert (await e.embed([TextPart(text="cat sat on mat")], "query"))[0] == a


async def test_hash_embedder_rejects_images():
    with pytest.raises(ValueError):
        await HashEmbedder().embed([ImagePart(data=b"x")], "query")


def test_supports():
    e = HashEmbedder()
    assert supports(e, TextPart(text="x"))
    assert not supports(e, ImagePart(data=b"x"))


def test_cosine_scores():
    m = normalize_rows(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32))
    s = cosine_scores([3.0, 0.0], m)
    assert s[0] == pytest.approx(1.0) and s[1] == pytest.approx(0.0)
    assert cosine_scores([0.0, 0.0], m).tolist() == [0.0, 0.0]


def test_registry():
    e = HashEmbedder(id="h1")
    reg = EmbedderRegistry([e])
    assert "h1" in reg and reg.get("h1") is e and reg.get("nope") is None
    reg.register(e)  # re-registering the same object is fine
    with pytest.raises(ValueError):
        reg.register(HashEmbedder(id="h1"))
    assert reg.ids() == ["h1"]


class CountingEmbedder:
    id = "count"
    modalities = {Modality.TEXT}
    dim = 2

    def __init__(self):
        self.calls = 0

    async def embed(self, items, purpose):
        self.calls += 1
        return [[float(len(i.text)), 1.0] for i in items]


async def test_cached_embedder_caches_query_text():
    inner = CountingEmbedder()
    c = CachedEmbedder(inner, maxsize=2)
    v1 = await c.embed([TextPart(text="ab")], "query")
    v2 = await c.embed([TextPart(text="ab")], "query")
    assert v1 == v2 == [[2.0, 1.0]] and inner.calls == 1
    await c.embed([TextPart(text="ab")], "document")  # documents bypass the cache
    assert inner.calls == 2
    assert c.id == "count" and c.dim == 2


@pytest.mark.slow
async def test_sentence_transformer_embedder():
    pytest.importorskip("sentence_transformers")
    from agentic_search.embedders.local import SentenceTransformerEmbedder
    e = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    [v] = await e.embed([TextPart(text="hello")], "query")
    assert len(v) == e.dim == 384
