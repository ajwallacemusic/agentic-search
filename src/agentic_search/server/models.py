"""HTTP request models and the lean response projection."""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agentic_search.core.result import SearchResult
from agentic_search.core.types import Budget, ImagePart, Query, TextPart
from agentic_search.events import SearchEvent, SearchFinished


class ImageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: str = Field(description="Standard or URL-safe base64 image bytes.")
    mime: str = Field(default="image/png", pattern=r"^image/[A-Za-z0-9.+-]+$")


class BudgetOverride(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_turns: int | None = Field(default=None, ge=1)
    max_tool_calls: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_cost_usd: float | None = Field(default=None, gt=0)
    max_seconds: float | None = Field(default=None, gt=0)


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: str | None = None
    question: str = Field(min_length=1, max_length=10_000)
    images: list[ImageInput] = Field(default_factory=list)
    sources: list[str] | None = None
    mode: Literal["retrieval", "harness", "model"] | None = None
    top_k: int = Field(default=20, ge=1, le=1000)
    budget: BudgetOverride | None = None
    include_content: bool = False
    include_trace: bool = False


class StreamRequest(SearchRequest):
    snapshot_k: int = Field(default=10, ge=0, le=100)


class RequestError(ValueError):
    """A well-formed request the service still cannot run (maps to HTTP 400)."""


def build_query(req: SearchRequest, *, max_images: int, max_image_bytes: int) -> Query:
    """Question text plus inline images. Images are always inline bytes: the service never
    accepts a `uri`, so a client cannot make the server read its own files."""
    if len(req.images) > max_images:
        raise RequestError(f"at most {max_images} images per request")
    parts: list[Any] = [TextPart(text=req.question)]
    for i, img in enumerate(req.images):
        # Reject empty data
        if not img.data:
            raise RequestError(f"images[{i}].data is empty")

        # Reject oversized encoded input before decoding (optimization + safety)
        # Base64 overhead: encoded size is roughly raw_size * 4/3
        # So if encoded_size > max_bytes * 4/3 + 4, reject pre-decode
        if len(img.data) > max_image_bytes * 4 // 3 + 4:
            raise RequestError(f"images[{i}] is larger than {max_image_bytes} bytes")

        # Convert URL-safe base64 to standard, add padding for unpadded input
        normalized = img.data.replace("-", "+").replace("_", "/")
        # Add padding to make length a multiple of 4 (only if needed)
        padding_needed = len(normalized) % 4
        if padding_needed:
            normalized += "=" * (4 - padding_needed)

        try:
            data = base64.b64decode(normalized, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RequestError(f"images[{i}].data is not valid base64") from exc

        # Check decoded size (post-decode safety check)
        if len(data) > max_image_bytes:
            raise RequestError(f"images[{i}] is larger than {max_image_bytes} bytes")

        parts.append(ImagePart(data=data, mime=img.mime))
    return Query(content=parts)


def resolve_budget(base: Budget, override: BudgetOverride | None,
                   ceilings: dict[str, float | int]) -> Budget:
    """The profile's budget, updated with the request's fields, then capped by the profile's
    ceilings. An unlimited (None) field under a ceiling becomes the ceiling."""
    fields = base.model_dump()
    if override is not None:
        fields.update(override.model_dump(exclude_none=True))
    for name, ceiling in ceilings.items():
        value = fields.get(name)
        if value is None or value > ceiling:
            fields[name] = ceiling
    return Budget(**fields)


def lean_result(result: SearchResult, *, include_content: bool, include_trace: bool) -> dict[str, Any]:
    """The service's JSON view of a result: no trace unless asked, no hit content unless asked."""
    data = result.model_dump(mode="json", exclude=None if include_trace else {"trace"})
    if not include_content:
        for ranked in data["hits"]:
            ranked["hit"].pop("content", None)
    return data


def render_event(event: SearchEvent, *, include_content: bool, include_trace: bool) -> str:
    """One event as a single-line JSON string; `search_finished` carries the lean result."""
    if isinstance(event, SearchFinished):
        data = event.model_dump(mode="json", exclude={"result"})
        data["result"] = lean_result(event.result, include_content=include_content,
                                     include_trace=include_trace)
        return json.dumps(data, separators=(",", ":"))
    return event.model_dump_json()
