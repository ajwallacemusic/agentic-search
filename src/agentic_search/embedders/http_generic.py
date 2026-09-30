"""Configurable HTTP embedder for custom model containers (e.g. MedSigLIP on Azure ML / AKS).

One request per item. `request` holds a JSON body template per modality; placeholders:
`{{text}}` (the text), `{{b64}}` (base64 image bytes), `{{mime}}` (image MIME type).
`response_path` is a small JSONPath (`$`, `.name`, `[n]`, `[*]`) whose FIRST match is the vector."""

from __future__ import annotations

import base64
import re
from typing import Any

from agentic_search.core.types import Content, ImagePart, Modality, image_bytes, text_of
from agentic_search.embedders.base import EmbedderError, Embedding, Purpose
from agentic_search.embedders.http import HttpEmbedder

_TOKEN = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)|\[(\*|\d+)\]|\[['\"]([^'\"]+)['\"]\]")
_PLACEHOLDER = re.compile(r"\{\{(?:text|b64|mime)\}\}")


def json_path(value: Any, path: str) -> list[Any]:
    """Evaluate a minimal JSONPath. Returns all matches (wildcards fan out)."""
    if not path.startswith("$"):
        raise ValueError(f"JSONPath must start with '$': {path!r}")
    rest, pos, steps = path[1:], 0, []
    while pos < len(rest):
        m = _TOKEN.match(rest, pos)
        if m is None:
            raise ValueError(f"unsupported JSONPath syntax at {rest[pos:]!r}")
        steps.append(m.group(1) or m.group(3) or m.group(2))
        pos = m.end()
    current = [value]
    for step in steps:
        nxt: list[Any] = []
        for node in current:
            if step == "*":
                if isinstance(node, list):
                    nxt.extend(node)
                elif isinstance(node, dict):
                    nxt.extend(node.values())
            elif step.isdigit() and isinstance(node, list):
                if int(step) < len(node):
                    nxt.append(node[int(step)])
            elif isinstance(node, dict) and step in node:
                nxt.append(node[step])
        current = nxt
    return current


def render(template: Any, values: dict[str, str]) -> Any:
    """Substitute placeholders in every string of a JSON template in a single pass, so a
    substituted value (e.g. user text containing `{{b64}}`) is never itself re-substituted."""
    if isinstance(template, str):
        return _PLACEHOLDER.sub(lambda m: values.get(m.group(0), m.group(0)), template)
    if isinstance(template, dict):
        return {k: render(v, values) for k, v in template.items()}
    if isinstance(template, list):
        return [render(v, values) for v in template]
    return template


class GenericHttpEmbedder(HttpEmbedder):
    def __init__(self, url: str, dim: int, *, request: dict[str, Any], response_path: str,
                 id: str | None = None, auth: Any = None, headers: dict[str, str] | None = None,
                 **kwargs: Any):
        modalities = set()
        if "text" in request:
            modalities.add(Modality.TEXT)
        if "image" in request:
            modalities.add(Modality.IMAGE)
        if not modalities:
            raise ValueError("request needs a 'text' and/or 'image' template")
        json_path({}, response_path)  # validate syntax early
        kwargs.setdefault("batch_size", 1)
        super().__init__(id or f"http:{url}", dim, modalities, auth=auth, **kwargs)
        self.url = url
        self.request = request
        self.response_path = response_path
        self.headers = headers or {}

    async def _embed_batch(self, batch: list[Content], purpose: Purpose) -> list[Embedding]:
        return [await self._one(item) for item in batch]

    async def _one(self, item: Content) -> Embedding:
        if isinstance(item, ImagePart):
            values = {"{{b64}}": base64.b64encode(image_bytes(item)).decode(), "{{mime}}": item.mime,
                      "{{text}}": ""}
            template = self.request["image"]
        else:
            values = {"{{text}}": text_of([item]), "{{b64}}": "", "{{mime}}": ""}
            template = self.request["text"]
        body = await self.post_json(self.url, render(template, values), self.headers)
        matches = json_path(body, self.response_path)
        if not matches or not isinstance(matches[0], list):
            raise EmbedderError(f"{self.id}: {self.response_path} matched no vector in the response")
        return matches[0]
