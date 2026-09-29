"""Local embedders: a zero-dependency hash embedder and sentence-transformers (text or CLIP)."""

from __future__ import annotations

import asyncio
import hashlib
import io
import re
from typing import Any

import numpy as np

from agentic_search.core.types import Content, ImagePart, Modality, image_bytes, text_of
from agentic_search.embedders.base import Embedding, Purpose

_TOKEN = re.compile(r"\w+")


class HashEmbedder:
    """Deterministic bag-of-words feature hashing. No downloads; for tests, demos and CI."""

    def __init__(self, dim: int = 256, id: str = "hash"):
        self.id = id
        self.dim = dim
        self.modalities = {Modality.TEXT}

    def _vector(self, text: str) -> Embedding:
        v = np.zeros(self.dim, dtype=np.float32)
        for token in _TOKEN.findall(text.lower()):
            v[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = float(np.linalg.norm(v))
        return (v / n if n else v).tolist()

    async def embed(self, items: list[Content], purpose: Purpose = "document") -> list[Embedding]:
        out = []
        for item in items:
            if isinstance(item, ImagePart):
                raise ValueError(f"{self.id} cannot embed images")
            out.append(self._vector(text_of([item])))
        return out


class SentenceTransformerEmbedder:
    """sentence-transformers model; set image=True for CLIP-style models. Needs the `local` extra."""

    def __init__(self, model_name: str, *, id: str | None = None, image: bool = False,
                 query_prefix: str = "", document_prefix: str = "", device: str | None = None):
        self.model_name = model_name
        self.id = id or f"st:{model_name}"
        self.modalities = {Modality.TEXT, Modality.IMAGE} if image else {Modality.TEXT}
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.device = device
        self._model: Any = None
        self._dim: int | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            d = self.model.get_sentence_embedding_dimension()
            self._dim = int(d) if d else len(self.model.encode(["x"])[0])
        return self._dim

    async def embed(self, items: list[Content], purpose: Purpose) -> list[Embedding]:
        return await asyncio.to_thread(self._encode, items, purpose)

    def _encode(self, items: list[Content], purpose: Purpose) -> list[Embedding]:
        prefix = self.query_prefix if purpose == "query" else self.document_prefix
        inputs: list[Any] = []
        for item in items:
            if isinstance(item, ImagePart):
                if Modality.IMAGE not in self.modalities:
                    raise ValueError(f"{self.id} is text-only")
                from PIL import Image
                inputs.append(Image.open(io.BytesIO(image_bytes(item))).convert("RGB"))
            else:
                inputs.append(prefix + text_of([item]))
        vectors = self.model.encode(inputs, normalize_embeddings=True, convert_to_numpy=True)
        return [v.tolist() for v in vectors]
