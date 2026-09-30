"""Gemini, and other Model Garden models, on Vertex AI through its OpenAI-compatible endpoint.

Needs the `openai` extra, and the `gcp` extra unless `auth` is supplied. The access token is
fetched per call through `auth`, so a caller with its own credentials (an async service with its
own transport) passes an `Auth` and the library never refreshes a token itself."""

from __future__ import annotations

import logging
import re
from typing import Any

from agentic_search.embedders.auth import Auth, GcpAdc
from agentic_search.models.openai_compat import OpenAICompatClient

logger = logging.getLogger(__name__)

_PROJECT = re.compile(r"[a-z][a-z0-9-]{4,28}[a-z0-9]")
_LOCATION = re.compile(r"[a-z]+(-[a-z0-9]+)*")


def vertex_base_url(project: str, location: str = "global") -> str:
    if not _PROJECT.fullmatch(project):
        raise ValueError(f"invalid Google Cloud project {project!r}")
    if not _LOCATION.fullmatch(location):
        raise ValueError(f"invalid Vertex location {location!r}")
    host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
    return f"https://{host}/v1/projects/{project}/locations/{location}/endpoints/openapi"


def vertex_client(model: str, *, project: str, location: str = "global", auth: Auth | None = None,
                  client: Any = None, supports_images: bool = True,
                  price_per_mtok: tuple[float, float] | None = None, id: str | None = None,
                  max_retries: int = 3) -> OpenAICompatClient:
    base_url = vertex_base_url(project, location)
    if price_per_mtok is None:
        logger.warning("vertex:%s has no price_per_mtok, so its cost will read 0 and a budget on "
                       "cost will not bind; pass the model's input and output price per million "
                       "tokens", model)
    publisher_model = model if "/" in model else f"google/{model}"
    # api_key is a placeholder the SDK requires; the Authorization header from `auth` replaces it.
    return OpenAICompatClient(publisher_model, base_url=base_url, api_key="unused", client=client,
                              supports_images=supports_images, price_per_mtok=price_per_mtok,
                              id=id or f"vertex:{model}", max_retries=max_retries,
                              auth=auth if auth is not None else GcpAdc())
