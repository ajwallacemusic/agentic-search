import asyncio

import anyio
import pytest

from agentic_search.server.sse import KEEPALIVE, format_sse, sse_body

from .conftest import lex, make_harness


def test_format_sse_frames_and_splits_lines():
    assert format_sse("phase_started", '{"a":1}', 4) == b'id: 4\nevent: phase_started\ndata: {"a":1}\n\n'
    assert format_sse("x", "l1\nl2") == b"event: x\ndata: l1\ndata: l2\n\n"


def slow_backend(docs_backend, entered, cancelled, delay=60):
    async def slow(op):
        entered.set()
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return []

    docs_backend.execute = slow


async def test_keepalive_during_silence(docs_backend):
    entered, cancelled = asyncio.Event(), asyncio.Event()
    slow_backend(docs_backend, entered, cancelled, delay=0.3)
    h = make_harness(docs_backend)
    chunks = [c async for c in sse_body(h.stream("q"), lambda e: e.model_dump_json(),
                                        keepalive_s=0.05)]
    assert chunks.count(KEEPALIVE) >= 3
    assert chunks[-1].startswith(b"id: ") and b"event: search_finished" in chunks[-1]


async def test_client_disconnect_cancels_search(docs_backend):
    entered, cancelled = asyncio.Event(), asyncio.Event()
    slow_backend(docs_backend, entered, cancelled)
    h = make_harness(docs_backend)
    gone = asyncio.Event()

    async def receive():
        await gone.wait()
        return {"type": "http.disconnect"}

    async def consume():
        chunks = []
        async for chunk in sse_body(h.stream("q"), lambda e: e.model_dump_json(),
                                    keepalive_s=10, receive=receive):
            chunks.append(chunk)
            if b"tool_call_started" in chunk:
                await entered.wait()
                gone.set()
        return chunks

    chunks = await asyncio.wait_for(consume(), 5)
    assert cancelled.is_set()
    assert not any(b"search_finished" in c for c in chunks)


async def test_closing_body_cancels_search(docs_backend):
    entered, cancelled = asyncio.Event(), asyncio.Event()
    slow_backend(docs_backend, entered, cancelled)
    h = make_harness(docs_backend)
    body = sse_body(h.stream("q"), lambda e: e.model_dump_json(), keepalive_s=10)

    async def consume():
        async for chunk in body:
            if b"tool_call_started" in chunk:
                await entered.wait()
                break
        await body.aclose()

    await asyncio.wait_for(consume(), 5)
    assert cancelled.is_set()


async def test_body_ends_after_terminal_event(docs_backend):
    h = make_harness(docs_backend, turns=[[lex("headache")]])
    chunks = [c async for c in sse_body(h.stream("q", mode="retrieval"),
                                        lambda e: e.model_dump_json(), keepalive_s=10)]
    assert b"event: search_finished" in chunks[-1]


@pytest.mark.parametrize("keepalive", [0.01])
async def test_keepalive_does_not_drop_events(docs_backend, keepalive):
    h = make_harness(docs_backend, turns=[[lex("headache"), lex("fever")]])
    chunks = [c async for c in sse_body(h.stream("q", mode="retrieval"),
                                        lambda e: e.model_dump_json(), keepalive_s=keepalive)]
    ids = [int(c.split(b"\n")[0][4:]) for c in chunks if c != KEEPALIVE]
    assert ids == list(range(len(ids)))


class _QueueStream:
    """Shaped like Harness._events: between yields it only awaits a queue."""

    def __init__(self):
        self.queue = asyncio.Queue()
        self.closed = False
        self._gen = self._run()

    async def _run(self):
        try:
            while True:
                yield await self.queue.get()
        finally:
            self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._gen.__anext__()

    async def aclose(self):
        await self._gen.aclose()


class _Event:
    type, seq = "x", 0


@pytest.mark.parametrize("event_ready_at_cancel", [False, True])
async def test_consumer_cancel_scope_always_closes_stream(event_ready_at_cancel):
    """Starlette cancels via an anyio scope (re-cancelling every await). The stream must
    still be closed deterministically, not left for garbage collection."""
    stream = _QueueStream()
    body = sse_body(stream, lambda e: "{}", keepalive_s=10)
    scopes = []

    async def consume():
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            async for _ in body:
                pass

    async with anyio.create_task_group() as tg:
        tg.start_soon(consume)
        await asyncio.sleep(0.05)            # body is waiting on the pending __anext__
        if event_ready_at_cancel:
            stream.queue.put_nowait(_Event())  # pending completes ...
        scopes[0].cancel()                     # ... in the same tick as the cancel
    await asyncio.sleep(0.05)
    assert stream.closed


async def test_watcher_exception_is_retrieved_when_body_is_closed(docs_backend):
    """A watcher that failed while the consumer held a frame is still marked retrieved, so no
    'Task exception was never retrieved' is logged when it is collected."""
    import gc

    loop = asyncio.get_running_loop()
    reported = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: reported.append(context))
    try:
        h = make_harness(docs_backend)
        boom = asyncio.Event()

        async def receive():
            await boom.wait()
            raise RuntimeError("receive failed")

        body = sse_body(h.stream("q"), lambda e: e.model_dump_json(), keepalive_s=10,
                        receive=receive)

        async def consume(body):
            await body.__anext__()  # search_started; the watcher is now running
            boom.set()
            for _ in range(5):
                await asyncio.sleep(0)  # the watcher fails while we hold the frame
            await body.aclose()

        await asyncio.wait_for(consume(body), 5)
        del body
        for _ in range(3):
            gc.collect()
            await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous)
    assert [c for c in reported if "never retrieved" in c.get("message", "")] == []
