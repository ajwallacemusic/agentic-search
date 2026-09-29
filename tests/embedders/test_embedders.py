import asyncio

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


async def test_cached_embedder_lru_eviction():
    """Test LRU eviction: with maxsize=2, embedding 'a','b','c' evicts 'a'; re-embedding 'a' triggers inner call."""
    inner = CountingEmbedder()
    c = CachedEmbedder(inner, maxsize=2)

    # Embed a, b, c as separate calls (3 texts but maxsize=2, so 'a' is evicted)
    await c.embed([TextPart(text="a")], "query")
    assert inner.calls == 1
    await c.embed([TextPart(text="b")], "query")
    assert inner.calls == 2
    await c.embed([TextPart(text="c")], "query")
    assert inner.calls == 3

    # At this point, cache has {b, c} and 'a' is evicted
    # Embedding 'c' again should use cache (no new call)
    await c.embed([TextPart(text="c")], "query")
    assert inner.calls == 3  # No new call

    # Embedding 'a' again should trigger inner call (it was evicted)
    await c.embed([TextPart(text="a")], "query")
    assert inner.calls == 4


async def test_cached_embedder_concurrent_safe():
    """Test concurrent safety: inner embedder that awaits; multiple coroutines should not raise KeyError."""
    class AsyncSlowEmbedder:
        id = "async_slow"
        modalities = {Modality.TEXT}
        dim = 2

        async def embed(self, items, purpose):
            await asyncio.sleep(0)  # Yield control to allow concurrent operations
            return [[float(len(i.text)), 1.0] for i in items]

    inner = AsyncSlowEmbedder()
    c = CachedEmbedder(inner, maxsize=1)

    # Run multiple concurrent embed calls with overlapping texts
    # With maxsize=1, this would cause eviction during awaits in the old implementation
    results = await asyncio.gather(
        c.embed([TextPart(text="x")], "query"),
        c.embed([TextPart(text="y")], "query"),
        c.embed([TextPart(text="x")], "query"),  # 'x' again
        c.embed([TextPart(text="y")], "query"),  # 'y' again
    )

    # Verify results are correct
    assert results[0] == [[1.0, 1.0]]  # 'x'
    assert results[1] == [[1.0, 1.0]]  # 'y'
    assert results[2] == [[1.0, 1.0]]  # 'x' again
    assert results[3] == [[1.0, 1.0]]  # 'y' again


async def test_cached_embedder_inner_wrong_count():
    """Test that ValueError is raised if inner embedder returns wrong number of vectors."""
    class WrongCountEmbedder:
        id = "wrong"
        modalities = {Modality.TEXT}
        dim = 2

        async def embed(self, items, purpose):
            # Always return 1 vector regardless of input count
            return [[1.0, 1.0]]

    inner = WrongCountEmbedder()
    c = CachedEmbedder(inner, maxsize=10)

    # Request embedding for 2 texts but inner returns only 1
    with pytest.raises(ValueError, match="returned 1 vectors for 2 texts"):
        await c.embed([TextPart(text="a"), TextPart(text="b")], "query")


@pytest.mark.slow
async def test_sentence_transformer_embedder():
    pytest.importorskip("sentence_transformers")
    from agentic_search.embedders.local import SentenceTransformerEmbedder
    e = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    [v] = await e.embed([TextPart(text="hello")], "query")
    assert len(v) == e.dim == 384
