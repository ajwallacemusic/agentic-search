"""Serve the Ignite demo over HTTP/SSE on localhost, plus the trace viewer at /.

Run: uv run python -m demo.ignite.serve [--port 8765]
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles

from agentic_search.core.harness import Harness
from agentic_search.core.secrets import scrub
from agentic_search.core.types import Document, text_of
from agentic_search.server import create_app
from agentic_search.server.config import AuthConfig, ServiceConfig
from demo.ignite.answer import answer_from, is_correct
from demo.ignite.build import build_demo_harness, load_config, make_client
from demo.ignite.check import Proof, load_proofs
from demo.ignite.corpus import Question, load_demo

VIEWER_DIR = Path(__file__).parent / "viewer"
PROFILE_OF = {"musique": "ignite", "wands": "ignite-wands"}  # corpus -> harness profile
HOST = "127.0.0.1"  # localhost only: the service has no auth


class AnswerRequest(BaseModel):
    id: str  # question id
    keys: list[str]  # a mode's top hits, best first ("source:doc_id")
    notes: list[str] = []  # facts the mode noted while searching (harness only)


def doc_texts(documents: dict[str, list[Document]]) -> dict[str, tuple[str, str]]:
    """"source:doc_id" -> (title, text), for the answer step."""
    return {f"{source}:{d.doc_id}": (str(d.metadata.get("title", "")), text_of(d.content))
            for source, docs in documents.items() for d in docs}


def create_demo_app(profiles: dict[str, Harness], questions: dict[str, Question], *,
                    viewer_dir: Path = VIEWER_DIR, docs: dict[str, tuple[str, str]] | None = None,
                    answer_client: Any = None, proofs: dict[str, Proof] | None = None) -> FastAPI:
    app = create_app(ServiceConfig(auth=AuthConfig(type="none"), default_profile="ignite"),
                     profiles)
    texts = docs or {}

    @app.get("/ignite/questions")
    async def list_questions() -> list[dict[str, str]]:
        return [{"id": q.id, "text": q.text, "kind": q.kind, "set": q.set,
                 "profile": PROFILE_OF[q.corpus]} for q in questions.values()]

    @app.post("/ignite/answer")
    async def answer(req: AnswerRequest) -> dict[str, Any]:
        """Called when a mode finishes: an answer read from its top documents, plus the gold
        documents (kept out of /ignite/questions, so nothing is given away while a run is live)."""
        q = questions.get(req.id)
        if q is None:
            raise HTTPException(404, f"unknown question {req.id!r}")
        out: dict[str, Any] = {"gold": q.gold, "expected": q.answers[0] if q.answers else None,
                               "answer": None, "steps": [], "correct": None, "error": None}
        out.update(_proof(proofs or {}, q.id, req.keys[:10], texts))
        picked = [texts[k] for k in req.keys[:10] if k in texts]
        if answer_client is None:
            out["error"] = "no answer model configured"
            return out
        try:
            out["answer"], out["steps"] = await answer_from(answer_client, q.text, picked,
                                                            [n[:500] for n in req.notes])
            out["correct"] = is_correct(out["answer"], q.answers)
        except Exception as exc:  # the answer line is extra; never fail the page over it
            out["error"] = scrub(f"{type(exc).__name__}: {exc}")[:200]
        return out

    app.mount("/", StaticFiles(directory=viewer_dir, html=True), name="viewer")
    return app


def _proof(proofs: dict[str, Proof], qid: str, keys: list[str],
           texts: dict[str, tuple[str, str]]) -> dict[str, Any]:
    """The rule behind the ★ marks, and per result its full text plus why it is (not) right."""
    proof = proofs.get(qid)
    if proof is None:
        return {"rule": None, "checks": {}, "coverage": None}
    checks = {}
    for k in keys:
        if k not in texts:  # e.g. an aggregate tool's count row: not a document at all
            checks[k] = {"verdict": "wrong", "why": "A tool's summary row, not a document. It "
                         "does not count.", "marks": [], "target": None, "title": "", "text": ""}
            continue
        title, text = texts[k]
        checks[k] = {**proof.check(k.split(":", 1)[-1]), "title": title, "text": text[:4000]}
    coverage = getattr(proof, "coverage", None)
    return {"rule": proof.rule(), "checks": checks,
            "coverage": coverage([k.split(":", 1)[-1] for k in keys]) if coverage else None}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, default=8765)
    return p


def main() -> None:
    import uvicorn

    args = build_parser().parse_args()
    cfg = load_config()
    data = load_demo()
    profiles: dict[str, Harness] = {}
    for corpus, corpus_data in data.items():
        profiles[PROFILE_OF[corpus]] = build_demo_harness(cfg, data=corpus_data)
        if os.environ.get("TYPESAFE_API_KEY"):
            profiles[f"{PROFILE_OF[corpus]}-typesafe"] = build_demo_harness(
                cfg, data=corpus_data, controller="typesafe")
    questions = {qid: q for d in data.values() for qid, q in d.questions.items()}
    docs = {k: v for d in data.values() for k, v in doc_texts(d.documents).items()}
    logging.basicConfig(level=logging.INFO)
    print(f"Ignite demo: http://{HOST}:{args.port}/  (profiles: {', '.join(profiles)})")
    # The answer box: one reader model, the same for both modes (config `answer`, else the judge).
    reader = make_client(cfg.get("answer") or cfg["judge"])
    app = create_demo_app(profiles, questions, docs=docs, answer_client=reader,
                          proofs=load_proofs(questions))
    uvicorn.run(app, host=HOST, port=args.port)


if __name__ == "__main__":
    main()
