"""Extension points for redaction, audit, and per-source model access policy."""

from __future__ import annotations

from typing import Any

from agentic_search.core.state import TraceEvent
from agentic_search.core.types import Hit


class Hooks:
    """Subclass to redact, audit, or block. Every payload bound for an external model or
    embedder passes through before_model_call; the returned value is what gets sent."""

    async def before_model_call(self, model_id: str, payload: Any) -> Any:
        return payload

    def on_trace_event(self, event: TraceEvent) -> None:
        return None


class SourcePolicy:
    """source -> model ids allowed to see that source's raw content. Unlisted sources are open."""

    def __init__(self, allowed: dict[str, set[str]] | None = None):
        self._allowed = {k: set(v) for k, v in (allowed or {}).items()}

    def allows(self, source: str, model_id: str) -> bool:
        allowed = self._allowed.get(source)
        return allowed is None or model_id in allowed

    def redact(self, hits: list[Hit], model_id: str) -> list[Hit]:
        return [h if self.allows(h.source, model_id)
                else h.model_copy(update={"content": [], "metadata": {**h.metadata, "_redacted": True}})
                for h in hits]
