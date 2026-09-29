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


async def test_cached_embedder_deduplicates_missing():
    """Test that duplicate uncached texts hit the inner embedder once."""
    inner = CountingEmbedder()
    c = CachedEmbedder(inner, maxsize=10)
    # Embed duplicate texts: ["a", "a"] should result in one inner call
    result = await c.embed([TextPart(text="a"), TextPart(text="a")], "query")
    assert result == [[1.0, 1.0], [1.0, 1.0]] and inner.calls == 1


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
    """Test concurrent safety: eviction race during await doesn't cause KeyError.

    Uses explicit synchronization to guarantee the race condition happens:
    - Pre-seeds cache with 'cached'
    - Call 1 awaits inner embed for 'new' then tries to read both texts
    - Call 2 starts during Call 1's await and adds 'evict' which evicts 'cached' due to maxsize=1

    With the old code, Call 1 would KeyError when trying to read evicted 'cached' from cache
    after the await and cache update. With the fix, it uses the local dict captured before
    await, so both 'cached' and 'new' are read from the local dict, not the shared cache.
    """
    call1_started = asyncio.Event()
    call1_can_proceed = asyncio.Event()

    class ControlledEmbedder:
        id = "controlled"
        modalities = {Modality.TEXT}
        dim = 2

        async def embed(self, items, purpose):
            # Only control Call 1's embed of 'new'; other calls proceed immediately
            texts = tuple(i.text for i in items)
            if texts == ("new",):
                # Call 1 is requesting 'new'; signal Call 2 and wait for permission
                call1_started.set()
                await call1_can_proceed.wait()
            else:
                # All other calls proceed immediately with a minimal sleep
                await asyncio.sleep(0)

            # Return distinguishing vectors based on text length and first char code
            return [[float(len(i.text)), float(ord(i.text[0]))] for i in items]

    inner = ControlledEmbedder()
    c = CachedEmbedder(inner, maxsize=1)

    # Pre-seed cache with 'cached'
    await c.embed([TextPart(text="cached")], "query")

    async def call1():
        # This will find 'cached' in cache and request 'new', which will block in the embedder
        return await c.embed([TextPart(text="cached"), TextPart(text="new")], "query")

    async def call2():
        # Wait for Call 1 to start awaiting
        await call1_started.wait()
        # Now issue a request that will cause eviction
        result = await c.embed([TextPart(text="evict")], "query")
        # Release Call 1 to proceed
        call1_can_proceed.set()
        return result

    results = await asyncio.gather(call1(), call2())

    # Call 1 should return correct vectors for 'cached' and 'new' (not KeyError)
    # 'cached': [6.0, 99.0], 'new': [3.0, 110.0]
    assert results[0] == [[6.0, 99.0], [3.0, 110.0]]
    # Call 2 should return correct vector for 'evict'
    # 'evict': [5.0, 101.0]
    assert results[1] == [[5.0, 101.0]]


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
