"""Hugging Face Text Embeddings Inference (TEI) `/embed` endpoint — the usual way to self-host
text embedders (BGE, E5, GTE, …). Text only."""

from __future__ import annotations

from typing import Any

from agentic_search.core.types import Content, Modality, text_of
from agentic_search.embedders.base import EmbedderError, Embedding, Purpose
from agentic_search.embedders.http import HttpEmbedder


class TEIEmbedder(HttpEmbedder):
    def __init__(self, url: str, dim: int, *, id: str | None = None, query_prefix: str = "",
                 document_prefix: str = "", normalize: bool = True, auth: Any = None, **kwargs: Any):
        super().__init__(id or f"tei:{url}", dim, {Modality.TEXT}, auth=auth, **kwargs)
        self.url = url.rstrip("/") + "/embed"
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.normalize = normalize

    async def _embed_batch(self, batch: list[Content], purpose: Purpose) -> list[Embedding]:
        prefix = self.query_prefix if purpose == "query" else self.document_prefix
        body = await self.post_json(self.url, {
            "inputs": [prefix + text_of([c]) for c in batch],
            "normalize": self.normalize, "truncate": True,
        })
        if not isinstance(body, list):
            raise EmbedderError(f"{self.id}: unexpected response shape")
        return body
