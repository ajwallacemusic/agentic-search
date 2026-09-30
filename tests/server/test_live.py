"""The service under a real uvicorn server: HTTP streaming and disconnect-cancels-search."""

import asyncio

import httpx
import uvicorn

from agentic_search.server import create_app

from .conftest import make_harness, parse_sse, service


async def serve(app):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning",
                                           lifespan="on"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, task, f"http://127.0.0.1:{port}"


async def test_live_stream_and_search(docs_backend):
    server, task, url = await serve(create_app(service(), {"demo": make_harness(docs_backend)}))
    try:
        async with httpx.AsyncClient(base_url=url) as c:
            async with c.stream("POST", "/v1/search/stream", json={"question": "q"}) as r:
                text = "".join([chunk async for chunk in r.aiter_text()])
            frames, _ = parse_sse(text)
            assert frames[0]["event"] == "search_started"
            assert frames[-1]["event"] == "search_finished"
            assert (await c.get("/healthz")).json()["status"] == "ok"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 10)


async def test_live_disconnect_cancels_backend_call(docs_backend):
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def slow(op):
        entered.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return []

    docs_backend.execute = slow
    app = create_app(service(keepalive_s=30), {"demo": make_harness(docs_backend)})
    server, task, url = await serve(app)
    try:
        async with httpx.AsyncClient(base_url=url) as c:
            async with c.stream("POST", "/v1/search/stream", json={"question": "q"}) as r:
                async for line in r.aiter_lines():
                    if line == "event: tool_call_started":
                        break
            await asyncio.wait_for(entered.wait(), 5)
        await asyncio.wait_for(cancelled.wait(), 5)
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 10)
