"""A tiny in-memory demo service: four medical/history documents, a BM25 echo driver and a
keyword judge. Used for the golden contract fixtures and for client end-to-end tests; needs no
network, models or credentials."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from agentic_search.backends.files import FilesBackend
from agentic_search.core.harness import Harness
from agentic_search.core.types import Document, TextPart
from agentic_search.embedders.local import HashEmbedder
from agentic_search.server.app import create_app
from agentic_search.server.config import AuthConfig, Profile, ServiceConfig
from agentic_search.testing import EchoDriver, KeywordJudge

DEMO_DOCS = [
    ("d1", "Aspirin reduces fever and relieves headache pain", {"title": "Aspirin"}),
    ("d2", "Ibuprofen is an anti-inflammatory used for pain", {"title": "Ibuprofen"}),
    ("d3", "The history of the printing press in Europe", {"title": "Printing press"}),
    ("d4", "Acetaminophen treats headache and fever", {"title": "Acetaminophen"}),
]


def demo_harness(*, slow_s: float | None = None,
                 on_cancel: Callable[[], None] | None = None) -> Harness:
    """`slow_s`: every backend call sleeps this long first; `on_cancel` runs if one is
    cancelled (lets tests observe that closing a stream cancels the search)."""
    docs = [Document(doc_id=i, content=[TextPart(text=t)], metadata=m) for i, t, m in DEMO_DOCS]
    backend = FilesBackend.from_documents("docs", docs, embedder=HashEmbedder())
    if slow_s is not None:
        execute = backend.execute

        async def slow_execute(op: Any) -> Any:
            try:
                await asyncio.sleep(slow_s)
            except asyncio.CancelledError:
                if on_cancel is not None:
                    on_cancel()
                raise
            return await execute(op)

        backend.execute = slow_execute  # type: ignore[method-assign]
    return Harness([backend], EchoDriver("docs", limit=10), embedders=[backend.embedder],
                   analyzer=KeywordJudge(["headache"]))


def demo_app(**profiles: Harness) -> Any:
    """The demo service with auth off. Extra keyword arguments add named profiles."""
    all_profiles = {"demo": Profile(demo_harness()),
                    **{name: Profile(h) for name, h in profiles.items()}}
    return create_app(ServiceConfig(auth=AuthConfig(type="none")), all_profiles)
