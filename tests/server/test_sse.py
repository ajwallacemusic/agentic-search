import asyncio

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
