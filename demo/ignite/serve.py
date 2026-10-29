"""Serve the Ignite demo over HTTP/SSE on localhost, plus the trace viewer at /.

Run: uv run python -m demo.ignite.serve [--port 8765]
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from fastapi import FastAPI
from starlette.staticfiles import StaticFiles

from agentic_search.core.harness import Harness
from agentic_search.server import create_app
from agentic_search.server.config import AuthConfig, ServiceConfig
from demo.ignite.build import build_demo_harness, load_config
from demo.ignite.corpus import Question, load_questions

VIEWER_DIR = Path(__file__).parent / "viewer"
HOST = "127.0.0.1"  # localhost only: the service has no auth


def create_demo_app(profiles: dict[str, Harness], questions: dict[str, Question], *,
                    viewer_dir: Path = VIEWER_DIR) -> FastAPI:
    app = create_app(ServiceConfig(auth=AuthConfig(type="none"), default_profile="ignite"),
                     profiles)

    @app.get("/ignite/questions")
    async def list_questions() -> list[dict[str, str]]:
        return [{"id": q.id, "text": q.text, "kind": q.kind} for q in questions.values()]

    app.mount("/", StaticFiles(directory=viewer_dir, html=True), name="viewer")
    return app


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, default=8765)
    return p


def main() -> None:
    import uvicorn

    args = build_parser().parse_args()
    cfg = load_config()
    profiles = {"ignite": build_demo_harness(cfg)}
    if os.environ.get("TYPESAFE_API_KEY"):
        profiles["ignite-typesafe"] = build_demo_harness(cfg, controller="typesafe")
    logging.basicConfig(level=logging.INFO)
    print(f"Ignite demo: http://{HOST}:{args.port}/  (profiles: {', '.join(profiles)})")
    uvicorn.run(create_demo_app(profiles, load_questions()), host=HOST, port=args.port)


if __name__ == "__main__":
    main()
