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
- **Analyzer decider**: judges relevance (`LLMJudge`, `CrossEncoderJudge`, `TypeSafeDecider`).
- **Controller decider**: continue/refine/broaden/stop (`LLMJudge`, `TypeSafeDecider`, or the built-in heuristic).
- **Backends**: files, Postgres + pgvector, MySQL, BigQuery, OpenSearch, Neo4j and Milvus (see below).
- **Hooks / SourcePolicy**: every model-bound payload passes through `Hooks.before_model_call`:
  the planner view, judge requests (question + hits), the controller view, embedder queries, and in
  model mode the delegate question/context and every tool output (a raising hook withholds that
  output). Per-source `allowed_models` withholds raw content from other models; each model's
  digest is rendered for that model.

## Backends

| type | install extra | lexical | vector | notes |
|---|---|---|---|---|
| `files` | – | BM25 | local index | directory or in-memory documents |
| `postgres` / `pgvector` | `postgres` | tsvector + `websearch_to_tsquery` | pgvector `<=>`/`<->`/`<#>` | read-only sessions, statement timeout |
| `mysql` | `mysql` | FULLTEXT (natural language) | – | lexical needs a FULLTEXT index; READ ONLY sessions, `max_execution_time` |
| `bigquery` | `bigquery` | term match (`CONTAINS_SUBSTR`) | `VECTOR_SEARCH` | every query dry-run; refused above `max_bytes_billed` |
| `opensearch` | `opensearch` | `multi_match` | k-NN (`knn_vector`) | search APIs only |
| `neo4j` | `neo4j` | FULLTEXT index | VECTOR index | collections are labels; `traverse`; READ sessions; native Cypher |
| `milvus` | `milvus` | BM25 function (sparse) | ANN | collections map 1:1; no regex/aggregate |

All backends support filters, regex, aggregates (`count`, `sum|avg|min|max:<column>`) and fetch.
Set `native_query: true` on a backend to let the planner run read-only native SQL / search bodies;
they pass `backends/native_guard.py` (single SELECT, no DML/DDL/locks/scripts, row cap) first.
Vector columns need `embedders: {<table>.<column>: <embedder id>}` so queries are embedded with the
same model as the stored vectors. DSNs and passwords are masked in errors and traces.
In a source with several tables/indices, hit ids are `<collection>/<primary key>`; tables with a
composite primary key are skipped unless `id_columns` names a column for them.

**Connect with a read-only role/user.** Grant the credentials you configure only `SELECT` (or
search/read) on the tables and indices you expose. The native-query guard and the read-only
sessions are defence in depth, not a substitute: the guard's function check is a denylist and
cannot anticipate every side-effecting function or extension.

```yaml
backends:
  - {name: notes, type: pgvector, dsn_env: NOTES_DSN, tables: [notes],
     embedders: {notes.embedding: "st:BAAI/bge-small-en-v1.5"}, native_query: true}
  - {name: orders, type: mysql, dsn_env: ORDERS_DSN}
  - {name: warehouse, type: bigquery, project: my-proj, dataset: clinical, max_bytes_billed: 500000000}
  - {name: search, type: opensearch, url_env: SEARCH_URL, indices: [articles]}
```

## Embedders

| type | modalities | notes |
|---|---|---|
| `hash`, `sentence_transformers` | text (+ image for CLIP) | in-process (`local`) |
| `openai_compat` | text | OpenAI, Azure OpenAI, vLLM, Ollama, Together |
| `tei` | text | Hugging Face Text Embeddings Inference |
| `vertex` | text; image with `multimodalembedding@001` | Google Application Default Credentials by default |
| `http` | per request template | custom containers, e.g. MedSigLIP on Azure ML |

Remote embedders take `auth: {type: api_key|bearer|gcp_adc|azure_identity, ...}` (install the `gcp` or
`azure` extra for the cloud credentials). Backends that embed their own documents send them to
non-local embedders only through `Hooks.before_model_call`, and `allowed_models` source policy applies to
embedders too.

```yaml
embedders:
  - {type: vertex, model: gemini-embedding-001, dim: 768, project: my-proj}
  - id: "azure:medsiglip-448"
    type: http
    url_env: MEDSIGLIP_URL
    dim: 1152
    auth: {type: azure_identity, scope: "api://medsiglip/.default"}
    request:
      image: {instances: [{image_b64: "{{b64}}"}]}
      text: {instances: [{text: "{{text}}"}]}
    response_path: "$.predictions[*].embedding"
controller: {type: typesafe, api_key_env: TYPESAFE_API_KEY}   # Jev judges/decides
```

## Evaluate

```bash
uv run python scripts/eval_beir.py --modes bm25 --limit 50                      # offline baseline
uv run python scripts/eval_beir.py --modes bm25,retrieval,harness --limit 50    # needs ANTHROPIC_API_KEY
```

`retrieval` is an unjudged baseline unless `--rerank-retrieval` is passed. Model mode returns only
the keys the model ranked, so its recall@100 is structurally lower than modes that return the pool.

## Develop

```bash
uv sync && uv run pytest                      # unit tests, no services
docker compose up -d --wait                   # Postgres+pgvector, MySQL, OpenSearch, Neo4j, Milvus
AGENTIC_SEARCH_INTEGRATION=1 uv run pytest -m integration   # backend contract suite
```
