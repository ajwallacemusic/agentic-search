"""Route a backend's document embeddings through Hooks and SourcePolicy.

Backends that embed their own content (FilesBackend) would otherwise send it to an embedder
without passing `Hooks.before_model_call`; with a remote embedder that is a leak. The Harness wraps
such embedders with HookedEmbedder via `Backend.bind_hooks` (see core/harness.py). Embedders that
declare `local = True` (in-process models) are left unwrapped: nothing leaves the machine."""

from __future__ import annotations

from typing import Any

from agentic_search.core.types import Content
from agentic_search.embedders.base import Embedder, EmbedderError, Embedding, Purpose


def is_hooked(embedder: Any) -> bool:
    """Check if embedder is hooked anywhere in its chain (including inside wrappers)."""
    current = embedder
    while current is not None:
        if getattr(current, "hooked", False):
            return True
        current = getattr(current, "inner", None)
    return False


def unwrap_hooked(embedder: Any) -> Any:
    """Return the innermost non-HookedEmbedder in the chain, or None if all are hooked."""
    if not isinstance(embedder, HookedEmbedder):
        return embedder
    return unwrap_hooked(embedder.inner)


class HookedEmbedder:
    hooked = True  # mark as hooked for detection in chains
    def __init__(self, inner: Embedder, hooks: Any, *, source: str, policy: Any):
        self.inner = inner
        self.hooks = hooks
        self.source = source
        self.policy = policy
        self.id = inner.id
        self.modalities = inner.modalities

    @property
    def dim(self) -> int:
        return self.inner.dim

    async def embed(self, items: list[Content], purpose: Purpose) -> list[Embedding]:
        if not self.policy.allows(self.source, self.id):
            raise EmbedderError(f"source policy forbids sending {self.source!r} content to "
                                f"embedder {self.id!r}")
        items = await self.hooks.before_model_call(self.id, items)
        return await self.inner.embed(items, purpose)
