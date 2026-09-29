# agentic-search

An agentic search harness: **plan → execute → analyze → decide** over any datastore, with swappable
models. Modeled on Doug Turnbull's *Three Kinds of Agentic Search* and SID.ai's SID-1.
Design: `docs/superpowers/specs/2026-09-29-agentic-search-harness-design.md`.

## Quickstart

```bash
uv sync --extra anthropic
export ANTHROPIC_API_KEY=...
```

```python
import asyncio
from agentic_search import Harness
from agentic_search.backends.files import FilesBackend
from agentic_search.embedders.local import HashEmbedder
from agentic_search.models.anthropic import AnthropicClient
from agentic_search.models.driver import ToolCallingDriver
from agentic_search.models.llm_judge import LLMJudge

async def main():
    client = AnthropicClient("claude-sonnet-5-5")
    emb = HashEmbedder()
    notes = FilesBackend("notes", "./my-notes", embedder=emb)
    async with Harness([notes], ToolCallingDriver(client), embedders=[emb],
                       analyzer=LLMJudge(client)) as h:
        result = await h.search("what did we decide about the Q3 launch?")
        for r in result.hits[:5]:
            print(f"{r.score:.2f} {r.hit.key} {r.hit.snippet(100)}")
        print(result.stop_reason, result.usage)

asyncio.run(main())
```

Or from YAML: `from agentic_search.config import load_harness`. The spec §7 shows the format.

## Modes

| mode | what happens |
|---|---|
| `retrieval` | one planned pass, rank by fused backend scores (judged if an analyzer is configured) |
| `harness` | full loop; the judge's feedback steers the next turn's plan |
| `model` | a trained search model (e.g. SID-1 via an OpenAI-compatible endpoint) runs its own tool loop; the harness enforces every budget (turns, tool calls, time, tokens, cost) and records the trace |

## Roles

- **Driver**: plans and calls tools (`ToolCallingDriver` over `AnthropicClient` or `OpenAICompatClient`).
- **Analyzer decider**: judges relevance (`LLMJudge`, `CrossEncoderJudge`, TypeSafe System One in Plan 3).
- **Controller decider**: continue/refine/broaden/stop (`LLMJudge`, or the built-in heuristic).
- **Backends**: `FilesBackend` now. SQL, OpenSearch, graph and vector stores come in Plans 2–3.
- **Hooks / SourcePolicy**: every model-bound payload passes through `Hooks.before_model_call`:
  the planner view, judge requests (question + hits), the controller view, embedder queries, and in
  model mode the delegate question/context and every tool output (a raising hook withholds that
  output). Per-source `allowed_models` withholds raw content from other models; each model's
  digest is rendered for that model.

## Evaluate

```bash
uv run python scripts/eval_beir.py --modes bm25 --limit 50                      # offline baseline
uv run python scripts/eval_beir.py --modes bm25,retrieval,harness --limit 50    # needs ANTHROPIC_API_KEY
```

`retrieval` is an unjudged baseline unless `--rerank-retrieval` is passed. Model mode returns only
the keys the model ranked, so its recall@100 is structurally lower than modes that return the pool.

## Develop

```bash
uv sync && uv run pytest
```
