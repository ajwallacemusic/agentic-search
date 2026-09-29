"""Compare harness modes on a BEIR dataset (spec success criterion 3).

Examples:
  uv run python scripts/eval_beir.py --modes bm25 --limit 50
  uv run --extra local python scripts/eval_beir.py --embedder st:BAAI/bge-small-en-v1.5 \
      --driver anthropic:claude-sonnet-5-5 --judge llm --modes bm25,retrieval,harness,model --limit 50
  uv run python scripts/eval_beir.py --driver openai:Qwen/Qwen3-8B --modes harness  # OPENAI_BASE_URL=...

The `retrieval` mode is an unjudged baseline (ranked by fused backend scores): --judge applies only
to harness and model modes unless --rerank-retrieval is given.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from agentic_search import Budget, Harness
from agentic_search.backends.files import FilesBackend
from agentic_search.eval.datasets import download_beir, load_beir
from agentic_search.eval.runner import run_eval
from agentic_search.testing import EchoDriver


def make_client(spec: str):
    provider, _, model = spec.partition(":")
    if provider == "anthropic":
        from agentic_search.models.anthropic import AnthropicClient
        return AnthropicClient(model)
    if provider == "openai":
        from agentic_search.models.openai_compat import OpenAICompatClient
        return OpenAICompatClient(model, base_url=os.environ.get("OPENAI_BASE_URL"))
    raise SystemExit(f"unknown driver provider {provider!r} (use anthropic:<model> or openai:<model>)")


def make_embedder(spec: str):
    if spec == "none":
        return None
    if spec == "hash":
        from agentic_search.embedders.local import HashEmbedder
        return HashEmbedder()
    if spec.startswith("st:"):
        from agentic_search.embedders.local import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(spec[3:])
    raise SystemExit(f"unknown embedder {spec!r} (none | hash | st:<model>)")


def make_judge(spec: str, client):
    if spec == "none":
        return None
    if spec == "llm":
        from agentic_search.models.llm_judge import LLMJudge
        return LLMJudge(client)
    if spec.startswith("cross_encoder"):
        from agentic_search.models.cross_encoder import CrossEncoderJudge
        _, _, model = spec.partition(":")
        return CrossEncoderJudge(model or "BAAI/bge-reranker-v2-m3")
    raise SystemExit(f"unknown judge {spec!r} (none | llm | cross_encoder[:model])")


def analyzer_for(mode: str, judge, *, rerank_retrieval: bool):
    """The retrieval baseline stays unjudged unless --rerank-retrieval is set."""
    return None if mode == "retrieval" and not rerank_retrieval else judge


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="nfcorpus")
    p.add_argument("--data-dir", default="data/beir")
    p.add_argument("--modes", default="bm25,retrieval,harness")
    p.add_argument("--driver", default="anthropic:claude-sonnet-5-5")
    p.add_argument("--judge", default="llm")
    p.add_argument("--rerank-retrieval", action="store_true",
                   help="also apply --judge to the retrieval mode (default: unjudged baseline)")
    p.add_argument("--embedder", default="none")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--max-turns", type=int, default=4)
    p.add_argument("--max-tool-calls", type=int, default=24)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--out", default="eval-results")
    args = p.parse_args()

    ds = load_beir(download_beir(args.dataset, args.data_dir))
    qids = sorted(ds.queries)[: args.limit]
    embedder = make_embedder(args.embedder)
    backend = FilesBackend.from_documents(
        "corpus", ds.documents(), embedder=embedder,
        description=f"BEIR {ds.name} corpus: one document per title + abstract.")
    budget = Budget(max_turns=args.max_turns, max_tool_calls=args.max_tool_calls, max_seconds=300)
    embedders = [embedder] if embedder else []
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = None
    for mode in args.modes.split(","):
        if mode == "bm25":
            harness = Harness([backend], EchoDriver("corpus"), mode="retrieval")
        else:
            from agentic_search.models.driver import ToolCallingDriver
            client = client or make_client(args.driver)
            analyzer = analyzer_for(mode, make_judge(args.judge, client),
                                    rerank_retrieval=args.rerank_retrieval)
            harness = Harness([backend], ToolCallingDriver(client), embedders=embedders,
                              analyzer=analyzer, mode=mode, budget=budget)
        report = await run_eval(harness, ds, name=f"{ds.name}/{mode}", query_ids=qids,
                                concurrency=args.concurrency)
        print(report.table(), flush=True)
        (out / f"{ds.name}-{mode}.json").write_text(report.model_dump_json(indent=1))


if __name__ == "__main__":
    asyncio.run(main())
