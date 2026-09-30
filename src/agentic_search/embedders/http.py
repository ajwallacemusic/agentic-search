"""Shared plumbing for remote embedders: batching, auth headers, retries with backoff, response
validation. Subclasses implement `_embed_batch` for one provider's request/response shape."""

from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx

from agentic_search.core.secrets import scrub
from agentic_search.core.types import Content, Modality, modality_of
from agentic_search.embedders.auth import NoAuth
from agentic_search.embedders.base import EmbedderError, Embedding, Purpose

RETRY_STATUSES = {408, 429, 500, 502, 503, 504, 529}


class HttpEmbedder:
    def __init__(self, id: str, dim: int, modalities: set[Modality], *, auth: Any = None,
                 batch_size: int = 32, concurrency: int = 4, timeout_s: float = 30.0,
                 max_retries: int = 4, backoff_s: float = 0.5, client: httpx.AsyncClient | None = None):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.id = id
        self.dim = dim
        self.modalities = modalities
        self.auth = auth or NoAuth()
        self.batch_size = max(1, batch_size)
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self._client = client
        self._owns_client = client is None
        self._sem = asyncio.Semaphore(max(1, concurrency))

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client

    async def post_json(self, url: str, payload: Any, extra_headers: dict[str, str] | None = None) -> Any:
        """POST JSON with auth headers; retry transient failures; raise EmbedderError otherwise."""
        client = self._get_client()
        last = ""
        for attempt in range(self.max_retries + 1):
            headers = {**(await self.auth.headers()), **(extra_headers or {})}
            try:
                resp = await client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code < 300:
                    try:
                        return resp.json()
                    except ValueError as exc:
                        raise EmbedderError(f"{self.id}: response is not JSON") from exc
                last = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if resp.status_code not in RETRY_STATUSES:
                    break
            if attempt < self.max_retries:
                await asyncio.sleep(self.backoff_s * (2 ** attempt) * (1 + random.random()))
        raise EmbedderError(scrub(f"{self.id}: {last}"))

    async def embed(self, items: list[Content], purpose: Purpose) -> list[Embedding]:
        for item in items:
            if modality_of(item) not in self.modalities:
                raise EmbedderError(f"{self.id} cannot embed {modality_of(item).value}")
        batches = [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]

        async def run(batch: list[Content]) -> list[Embedding]:
            async with self._sem:
                return await self._embed_batch(batch, purpose)

        results = await asyncio.gather(*(run(b) for b in batches))
        vectors = [v for batch in results for v in batch]
        if len(vectors) != len(items):
            raise EmbedderError(f"{self.id}: expected {len(items)} vectors, got {len(vectors)}")
        for v in vectors:
            if len(v) != self.dim:
                raise EmbedderError(f"{self.id}: expected dim {self.dim}, got {len(v)}")
        return [[float(x) for x in v] for v in vectors]

    async def _embed_batch(self, batch: list[Content], purpose: Purpose) -> list[Embedding]:
        raise NotImplementedError

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None
        close = getattr(self.auth, "close", None)
        if close is not None:
            await close()
