"""Embedder protocol, registry, query cache, and vector math helpers."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Literal, Protocol, runtime_checkable

import numpy as np

from agentic_search.core.types import Content, Modality, TextPart, modality_of

Embedding = list[float]
Purpose = Literal["query", "document"]


@runtime_checkable
class Embedder(Protocol):
    id: str
    modalities: set[Modality]

    @property
    def dim(self) -> int: ...

    async def embed(self, items: list[Content], purpose: Purpose) -> list[Embedding]: ...


def supports(embedder: Embedder, part: Content) -> bool:
    return modality_of(part) in embedder.modalities


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine_scores(query: Embedding, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of `query` against each row. Rows must already be L2-normalized."""
    q = np.asarray(query, dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n == 0.0:
        return np.zeros(len(matrix), dtype=np.float32)
    return matrix @ (q / n)


class EmbedderRegistry:
    """Maps embedder ids (as recorded on manifest vector fields) to embedders."""

    def __init__(self, embedders: Iterable[Embedder] = ()):
        self._by_id: dict[str, Embedder] = {}
        for e in embedders:
            self.register(e)

    def register(self, embedder: Embedder) -> None:
        existing = self._by_id.get(embedder.id)
        if existing is not None and existing is not embedder:
            raise ValueError(f"duplicate embedder id {embedder.id!r}")
        self._by_id[embedder.id] = embedder

    def get(self, embedder_id: str) -> Embedder | None:
        return self._by_id.get(embedder_id)

    def __contains__(self, embedder_id: object) -> bool:
        return embedder_id in self._by_id

    def ids(self) -> list[str]:
        return sorted(self._by_id)


class CachedEmbedder:
    """LRU cache for text query embeddings; planners often re-embed similar rewrites."""

    def __init__(self, inner: Embedder, maxsize: int = 1024):
        self.inner = inner
        self.id = inner.id
        self.modalities = inner.modalities
        self.maxsize = maxsize
        self._cache: OrderedDict[str, Embedding] = OrderedDict()

    @property
    def dim(self) -> int:
        return self.inner.dim

    async def embed(self, items: list[Content], purpose: Purpose) -> list[Embedding]:
        if purpose != "query" or not all(isinstance(i, TextPart) for i in items):
            return await self.inner.embed(items, purpose)
        texts = [i.text for i in items]  # type: ignore[union-attr]
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if missing:
            vectors = await self.inner.embed([TextPart(text=t) for t in missing], "query")
            self._cache.update(zip(missing, vectors))
        out = []
        for t in texts:
            self._cache.move_to_end(t)
            out.append(self._cache[t])
        while len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
        return out
