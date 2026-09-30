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

## Streaming

`Harness.stream()` runs one search and yields typed events as it goes, so an application can
show progress (planning, querying each source, judging, deciding) and a result list that grows
turn by turn:

```python
from agentic_search import PhaseStarted, ResultsUpdated, SearchFailed, SearchFinished
from agentic_search.config import load_harness

async with load_harness("search.yaml") as h:
    async with h.stream("what treats headache?", snapshot_k=10) as events:
        async for ev in events:
            match ev:
                case PhaseStarted(phase=phase):
                    print("…", phase)
                case ResultsUpdated(hits=hits):
                    print([s.key for s in hits])
                case SearchFinished(result=result):
                    print(result.keys())
                case SearchFailed(error_type=kind, message=msg):
                    print("failed:", kind, msg)
```

- Every event has `type`, `schema_version`, `search_id`, `seq` (0, 1, 2, … with no gaps),
  `turn` and `at_ms`. Events are Pydantic models; `ev.model_dump_json()` is ready for SSE.
- The last event is always `search_finished` or `search_failed`, unless the stream is cancelled.
  Only an invalid `mode`, `top_k` or `snapshot_k` raises `HarnessError` from `stream()` itself;
  unknown sources and every other failure end the stream with `search_failed`.
- Event `at_ms` counts from the start of the stream; trace `at_ms` counts from the trace's own
  start, so the two are not comparable.
- Leaving the `async with` block or cancelling the consuming task cancels the search, including
  in-flight backend and model calls.
- Snapshots carry a scrubbed 300-character snippet per hit; pass `include_content=True` for full
  content, or `snapshot_k=0` to turn them off.
- `search()` is `stream()` drained to its final result. Consumers must ignore unknown event types.

## Service

`pip install 'agentic-search[server]'` adds an HTTP service that streams searches as
server-sent events. Operators define named **profiles** (each a harness config) at startup;
clients pick a profile and override per request, but never send credentials or datastores.

```yaml
# service.yaml
service:
  auth: {type: api_key, keys_env: SEARCH_API_KEYS}   # or {type: none}, which must be explicit
  cors_origins: ["https://app.example.com"]
  max_concurrent_searches: 16
  default_profile: general
  keepalive_s: 15              # SSE keep-alive comment after this much silence
  max_image_bytes: 10000000    # per image, decoded
  max_images: 4                # per request
  max_body_bytes: 53398869     # default: room for max_images base64 images + 64 KiB
  setup_retry_s: 30            # a profile whose setup failed is retried after this long
  expose_docs: false           # true serves /docs, /redoc and /openapi.json
profiles:
  general:
    backends: [{name: notes, type: files, root: ./docs, glob: "**/*.md"}]
    driver: {type: anthropic, model: claude-sonnet-5-5}
  strict:
    backends: [{name: notes, type: files, root: ./docs, glob: "**/*.md"}]
    driver: {type: anthropic, model: claude-sonnet-5-5}
    limits: {max_budget: {max_turns: 2, max_cost_usd: 0.25}, allow_include_content: false}
```

```bash
SEARCH_API_KEYS=key1,key2 agentic-search serve --config service.yaml --port 8080
```

`serve` also takes `--host`, `--log-level` and `--graceful-timeout` (seconds open searches get
on shutdown, default 10). A config error prints `agentic-search: <message>` and exits 2. API
keys must be printable ASCII.

| Endpoint | |
|---|---|
| `GET /healthz` | Liveness, no auth: `{"status": "ok" \| "degraded", "version": "<version>"}` |
| `GET /v1/profiles` | Profiles with their sources, capabilities, budget and limits |
| `POST /v1/search` | One-shot search; returns the result as JSON |
| `POST /v1/search/stream` | The same search as `text/event-stream` |

Request body: `{profile?, question, images?: [{data: <base64>, mime}], sources?, mode?, top_k?,
budget?, include_content?, include_trace?}`, plus `snapshot_k?` for the stream. Send the key as
`Authorization: Bearer <key>` or `X-API-Key`. The key is checked before the body is read.

- Each SSE frame is `id: <seq>`, `event: <type>`, `data: <event JSON>`; a `: keep-alive`
  comment is sent after `keepalive_s` (default 15 s) of silence.
- The final `search_finished` carries a lean result: no trace unless `include_trace`, no hit
  content unless `include_content` (and the profile allows it). With
  `allow_include_content: false`, `results_updated` snapshots still carry each hit's text
  snippet and results still carry the judge's rationales; only the full `content` is withheld.
- Question images are not echoed back: in `search_started`, `search_finished` and the
  `/v1/search` body, each question image part keeps `kind` and `mime` but has `data: null`.
- A request may lower any budget field. Each field is capped by the profile's
  `limits.max_budget` if set there, otherwise by the profile's own budget: to let clients raise
  a field, set `max_budget` for it explicitly.
- Closing the connection cancels the search, including in-flight backend and model calls, on
  both `/v1/search` and `/v1/search/stream`.
- Status codes:
  - `400`: a request the service cannot run. Examples: `include_content` on a profile that
    forbids it, invalid base64 or too many or too-large images, no `profile` when there are several
    profiles and no default, and unknown `sources` on `/v1/search`.
  - `401`: missing or bad key.
  - `404`: unknown profile.
  - `413`: body over `max_body_bytes`.
  - `422`: invalid body.
  - `429`: too many concurrent searches.
  - `503`: profile unavailable (setup is retried after `setup_retry_s`).

  Failures during a streamed search, such as unknown `sources`, arrive as a `search_failed`
  event.
- Images are always inline base64; the service never reads a client-supplied path or URI.
- `agentic-search export-schema --out schema/` writes the event JSON Schema and golden SSE
  fixtures that clients test against (committed under `schema/`).
- A TypeScript client lives in [`clients/typescript`](clients/typescript/README.md).

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

Every backend supports filters and fetch. All except Milvus also support regex and aggregates
(`count`, `sum|avg|min|max:<column>`); `traverse` is Neo4j-only.
Set `native_query: true` on a backend to let the planner run read-only native SQL / search bodies;
they pass `backends/native_guard.py` (single SELECT, no DML/DDL/locks/scripts, row cap) first.
Vector columns need `embedders: {<table>.<column>: <embedder id>}` so queries are embedded with the
same model as the stored vectors. DSNs and passwords are masked in errors and traces.
Milvus never loads collections by default (reads fail with "collection … is not loaded"); set
`load_collections: true` to let the backend call `load_collection` on first use.
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
