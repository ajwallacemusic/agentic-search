"""Runs the demo service on a free port for the TypeScript end-to-end tests.

Prints `LISTENING <port>` once serving, and `CANCELLED` whenever a backend call of the `slow`
profile is cancelled (i.e. a client closed its stream). Run with `uv run python serve_demo.py`."""

import asyncio

import uvicorn

from agentic_search.server.demo import demo_app, demo_harness


def cancelled() -> None:
    print("CANCELLED", flush=True)


async def main() -> None:
    app = demo_app(slow=demo_harness(slow_s=30, on_cancel=cancelled))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    print(f"LISTENING {server.servers[0].sockets[0].getsockname()[1]}", flush=True)
    await task


if __name__ == "__main__":
    asyncio.run(main())
