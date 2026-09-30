"""Server-sent events framing, keep-alives and client-disconnect handling for one search."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Awaitable, Callable

from agentic_search.core.harness import SearchStream
from agentic_search.events import SearchEvent

KEEPALIVE = b": keep-alive\n\n"


def format_sse(event_type: str, data: str, event_id: int | None = None) -> bytes:
    lines = [] if event_id is None else [f"id: {event_id}"]
    lines.append(f"event: {event_type}")
    lines.extend(f"data: {line}" for line in data.split("\n"))
    return ("\n".join(lines) + "\n\n").encode()


async def wait_for_disconnect(receive: Callable[[], Awaitable[dict[str, Any]]]) -> None:
    """Return once the ASGI `receive` channel reports `http.disconnect`."""
    while True:
        message = await receive()
        if message.get("type") == "http.disconnect":
            return


async def sse_body(stream: SearchStream, render: Callable[[SearchEvent], str], *,
                   keepalive_s: float,
                   receive: Callable[[], Awaitable[dict[str, Any]]] | None = None,
                   ) -> AsyncIterator[bytes]:
    """Frame every event of `stream` as SSE, sending a keep-alive comment after `keepalive_s`
    of silence. If the client disconnects (seen via `receive`), the search is cancelled."""
    iterator = stream.__aiter__()
    pending: asyncio.Future[SearchEvent] | None = None
    watcher = asyncio.ensure_future(wait_for_disconnect(receive)) if receive else None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(iterator.__anext__())
            waiting = {pending} if watcher is None else {pending, watcher}
            done, _ = await asyncio.wait(waiting, timeout=keepalive_s,
                                         return_when=asyncio.FIRST_COMPLETED)
            if watcher is not None and watcher in done:
                if not watcher.cancelled():
                    watcher.exception()
                return  # client went away: the finally block cancels the search
            if pending not in done:
                yield KEEPALIVE
                continue
            finished, pending = pending, None
            try:
                event = finished.result()
            except StopAsyncIteration:
                return
            yield format_sse(event.type, render(event), event.seq)
    finally:
        if watcher is not None:
            watcher.cancel()
        if pending is not None and not pending.done():
            # In flight: its CancelledError runs the stream's own finally, which cancels the
            # search. wait() never raises the future's error; our own cancellation may still
            # interrupt it, and that is fine because the search is already being cancelled.
            pending.cancel()
            await asyncio.wait({pending})
        if pending is not None and pending.done() and not pending.cancelled():
            pending.exception()  # mark retrieved; the stream is being abandoned anyway
        # Nothing in flight now. aclose() cancels the search synchronously before its first
        # await, so the search stops even if this await is itself re-cancelled.
        await stream.aclose()
