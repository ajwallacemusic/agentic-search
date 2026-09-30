"""Vertex AI embeddings via the `:predict` endpoint.

- Text models (`gemini-embedding-001`, `text-embedding-005`, …): instances
  `{"content": text, "task_type": RETRIEVAL_QUERY|RETRIEVAL_DOCUMENT}`, response
  `predictions[i].embeddings.values`.
- Multimodal (`multimodalembedding@001`): one instance per request,
  `{"text": …}` or `{"image": {"bytesBase64Encoded": …}}`, response
  `predictions[0].textEmbedding` / `imageEmbedding`; text and images share one space.

Auth defaults to Google Application Default Credentials (the `gcp` extra)."""

from __future__ import annotations

import base64
from typing import Any

from agentic_search.core.types import Content, ImagePart, Modality, image_bytes, text_of
from agentic_search.embedders.auth import GcpAdc
from agentic_search.embedders.base import EmbedderError, Embedding, Purpose
from agentic_search.embedders.http import HttpEmbedder

TASK_TYPES = {"query": "RETRIEVAL_QUERY", "document": "RETRIEVAL_DOCUMENT"}


class VertexEmbedder(HttpEmbedder):
    def __init__(self, model: str, dim: int, *, project: str, location: str = "us-central1",
                 auth: Any = None, id: str | None = None, batch_size: int | None = None,
                 endpoint: str | None = None, **kwargs: Any):
        self.multimodal = model.startswith("multimodalembedding")
        if batch_size is None:
            batch_size = 1 if self.multimodal or model.startswith("gemini-embedding") else 32
        modalities = {Modality.TEXT, Modality.IMAGE} if self.multimodal else {Modality.TEXT}
        super().__init__(id or f"vertex:{model}", dim, modalities, auth=auth or GcpAdc(),
                         batch_size=batch_size, **kwargs)
        self.model = model
        host = endpoint or f"https://{location}-aiplatform.googleapis.com"
        self.url = (f"{host}/v1/projects/{project}/locations/{location}"
                    f"/publishers/google/models/{model}:predict")

    async def _embed_batch(self, batch: list[Content], purpose: Purpose) -> list[Embedding]:
        if self.multimodal:
            return [await self._multimodal(item) for item in batch]
        body = await self.post_json(self.url, {
            "instances": [{"content": text_of([c]), "task_type": TASK_TYPES[purpose]} for c in batch],
            "parameters": {"outputDimensionality": self.dim, "autoTruncate": True},
        })
        try:
            return [p["embeddings"]["values"] for p in body["predictions"]]
        except (KeyError, TypeError) as exc:
            raise EmbedderError(f"{self.id}: unexpected response shape") from exc

    async def _multimodal(self, item: Content) -> Embedding:
        if isinstance(item, ImagePart):
            instance: dict[str, Any] = {"image": {"bytesBase64Encoded":
                                                  base64.b64encode(image_bytes(item)).decode()}}
            key = "imageEmbedding"
        else:
            instance, key = {"text": text_of([item])}, "textEmbedding"
        body = await self.post_json(self.url, {"instances": [instance],
                                               "parameters": {"dimension": self.dim}})
        try:
            return body["predictions"][0][key]
        except (KeyError, IndexError, TypeError) as exc:
            raise EmbedderError(f"{self.id}: unexpected response shape") from exc
