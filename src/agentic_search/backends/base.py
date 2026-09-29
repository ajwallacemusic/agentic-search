"""Backend protocol: every datastore adapter implements this."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from agentic_search.core.types import Capability, Hit, Manifest, QueryOp

DiscoverDetail = Literal["summary", "full"]


class BackendError(Exception):
    """Raised by adapters for query failures; the executor turns it into a ToolError."""


class UnsupportedOperation(BackendError):
    """The adapter does not implement this op (or this variant of it)."""


@runtime_checkable
class Backend(Protocol):
    name: str
    backend_type: str

    def capabilities(self) -> set[Capability]: ...

    async def discover(self, detail: DiscoverDetail = "full",
                       collection: str | None = None) -> Manifest: ...

    async def execute(self, op: QueryOp) -> list[Hit]: ...

    async def close(self) -> None: ...


def rrf_merge(rankings: list[tuple[list[str], float]], k: int = 60) -> list[tuple[str, float]]:
    """Weighted reciprocal rank fusion. rankings = [(ids_best_first, weight), ...]."""
    scores: dict[str, float] = {}
    for ids, weight in rankings:
        for rank, doc_id in enumerate(ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
