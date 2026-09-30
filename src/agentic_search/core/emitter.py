"""Per-search event emitter: stamps the common fields, scrubs payloads, hands events to a sink."""

from __future__ import annotations

import time
from typing import Any, Callable, TypeVar

from agentic_search.core.secrets import scrub_data

E = TypeVar("E")


class EventEmitter:
    """`emit` is synchronous, so `seq` stays gap-free and ordered even with concurrent tool calls
    on one event loop."""

    def __init__(self, search_id: str, sink: Callable[[Any], None], *,
                 clock: Callable[[], float] = time.monotonic):
        self.search_id = search_id
        self._sink = sink
        self._clock = clock
        self._t0 = clock()
        self._seq = 0

    def emit(self, cls: type[E], *, turn: int, **fields: Any) -> E:
        event = cls(search_id=self.search_id, seq=self._seq, turn=turn,  # type: ignore[call-arg]
                    at_ms=(self._clock() - self._t0) * 1000, **scrub_data(fields))
        self._seq += 1
        self._sink(event)
        return event


class NullEmitter(EventEmitter):
    """Discards events; the default on SearchState so roles work without a stream."""

    def __init__(self) -> None:
        super().__init__("", lambda _event: None)
