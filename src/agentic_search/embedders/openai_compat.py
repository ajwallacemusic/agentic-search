"""OpenAI-compatible `/embeddings` endpoint: OpenAI, Azure OpenAI (via base_url + api-key auth),
vLLM, Ollama, Together, and most hosted open models. Text only."""

from __future__ import annotations

from typing import Any

from agentic_search.core.types import Content, Modality, text_of
from agentic_search.embedders.auth import Bearer
from agentic_search.embedders.base import EmbedderError, Embedding, Purpose
from agentic_search.embedders.http import HttpEmbedder


class OpenAICompatEmbedder(HttpEmbedder):
    def __init__(self, model: str, dim: int, *, base_url: str = "https://api.openai.com/v1",
                 api_key: str | None = None, auth: Any = None, dimensions: int | None = None,
                 query_prefix: str = "", document_prefix: str = "", id: str | None = None,
                 **kwargs: Any):
        if auth is None and api_key:
            auth = Bearer(api_key)
        super().__init__(id or f"openai:{model}", dim, {Modality.TEXT}, auth=auth, **kwargs)
        self.model = model
        self.url = base_url.rstrip("/") + "/embeddings"
        self.dimensions = dimensions
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

    async def _embed_batch(self, batch: list[Content], purpose: Purpose) -> list[Embedding]:
        prefix = self.query_prefix if purpose == "query" else self.document_prefix
        payload: dict[str, Any] = {"model": self.model, "input": [prefix + text_of([c]) for c in batch]}
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        body = await self.post_json(self.url, payload)
        try:
            rows = sorted(body["data"], key=lambda r: r["index"])
            return [r["embedding"] for r in rows]
        except (KeyError, TypeError) as exc:
            raise EmbedderError(f"{self.id}: unexpected response shape") from exc
