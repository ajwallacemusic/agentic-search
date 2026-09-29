# Agentic Search Harness — Design Spec

- **Date:** 2026-09-29
- **Status:** Draft for review
- **Scope:** Sub-project 1 of 4 (core harness + v1 adapters + minimal eval)

## 1. Purpose

A Python library that performs *agentic search*: given a question and a set of
data sources, it plans searches, executes them against any underlying datastore,
analyzes the results, and decides what to do next, until it returns a ranked
list of relevant documents with evidence and a full trace.

It is modeled on two references:

- Doug Turnbull, *Three Kinds of Agentic Search* (softwaredoug.com, 2026-06-08):
  retrieval-centric, harness-centric (judge feedback in the loop), and
  model-centric (a trained search model drives simple backends).
- SID.ai **SID-1**: a search subagent that takes a question, tools, and a
  dataset; issues many parallel searches per turn (BM25, dense ANN, metadata
  filters, regex/glob, HyDE) over ~3–4 turns; returns a ranked document list.

The harness supports all three kinds as configurable modes over one codebase so
they can be compared on the same eval set.

### Goals

1. Every search call goes through a backend interface that can point at any
   datastore (OpenSearch, pgvector, Milvus, Neo4j, BigQuery, MySQL, plain files;
   Neptune and others later).
2. Every model is swappable. Planning and tool-calling (drivers) and typed
   judgments (deciders) are separate roles, so System One decision models such
   as TypeSafe Jev/Laya, cross-encoders, and LLM judges all fit.
3. Embedders are swappable and may be remote (Vertex AI Gemini, self-hosted
   MedSigLIP on Azure, etc.), with protection against embedding-space mismatch.
4. Modality-general content model (text, image, structured) from day one.
5. Ready for a later medical specialization (PHI controls, ontologies) through
   hooks, without building it now.

### Non-goals (v1)

MCP server, HTTP API, CLI; Neptune adapter; answer synthesis; the medical
specialization; PHI redaction itself (only the hooks); write operations of any
kind against datastores.

### Decomposition (later sub-projects)

2. Additional backend adapters (Neptune, others) as plugins.
3. Full evaluation suite (more datasets, dashboards, regression tracking).
4. Medical specialization: ontologies (UMLS/SNOMED/MeSH/ICD-10), domain
   judges, PHI redaction and audit, source-level access policy.

## 2. Decisions

| Topic | Decision |
|---|---|
| Language | Python 3.11+, async throughout, Pydantic v2 models |
| Output contract | Ranked hits + evidence + full trace (SID-1 style); no answer synthesis |
| Interface | Python library only in v1 |
| Loop architecture | Explicit Plan → Execute → Analyze → Decide state machine with pluggable roles (Approach A) |
| Query model | Typed operations + shared Filter AST; optional read-only `native_query` escape hatch, disabled by default |
| v1 backends | files (+ in-memory BM25/vector), OpenSearch, Postgres+pgvector, Neo4j, Milvus, BigQuery, MySQL |
| v1 drivers | Anthropic Claude; OpenAI-compatible endpoint (vLLM, Ollama, Together, Baseten/SID-1, OpenAI) |
| v1 deciders | TypeSafe System One (Jev/Laya etc.); local cross-encoder/reranker; LLM judge (any driver model) |
| v1 embedders | Vertex AI (Gemini text + multimodal); OpenAI-compatible `/embeddings`; HF TEI; generic HTTP (configurable); local sentence-transformers/CLIP |

## 3. Architecture

### 3.1 Package layout

```
agentic_search/
  core/
    types.py        # Content, Document, Hit, QueryOp union, Filter AST, Manifest, Capability
    state.py        # SearchState, CandidatePool, Budget, Trace + TraceEvent
    harness.py      # Harness: the loop
    modes.py        # retrieval | harness | model(delegate) presets
    hooks.py        # hook protocol + no-op defaults
  backends/
    base.py         # Backend protocol, shared caps/limit enforcement
    filters.py      # Filter AST → OpenSearch DSL / SQL (pg, mysql, bigquery) / Cypher / Milvus expr
    native_guard.py # read-only validation for native_query
    files.py  opensearch.py  pgvector.py  neo4j.py  milvus.py  bigquery.py  mysql.py
  embedders/
    base.py         # Embedder protocol, batching, retry, cache
    auth.py         # ApiKey, Bearer, GcpAdc, AzureIdentity
    vertex.py  openai_compat.py  tei.py  http_generic.py  local.py
  models/
    base.py         # Driver, Decider protocols; ToolCall, Judgment, Decision
    anthropic.py  openai_compat.py  typesafe.py  cross_encoder.py  llm_judge.py
  roles/
    planner.py      # default Planner (wraps a Driver)
    executor.py     # validation + parallel execution + pooling
    analyzer.py     # judging + fusion + digest
    controller.py   # stop/continue logic + budget enforcement
  eval/
    datasets.py     # BEIR-format loader (queries, corpus, qrels)
    metrics.py      # NDCG@k, Recall@k, MRR
    runner.py       # run modes/configs, report quality + cost + latency
```

Each module does one thing and talks to the others only through the protocols
in `*/base.py` and the types in `core/types.py`.

### 3.2 Core types (`core/types.py`)

- `Content` is a tagged union of `TextPart(text)`, `ImagePart(uri | bytes, mime)`
  and `StructuredPart(data: dict)`. Queries and hits both carry `list[Content]`.
- `Hit` has `doc_id`, `source`, `content: list[Content]`, `metadata: dict`,
  `raw_score: float | None`, and `provenance: list[OpRef]` (every op that
  retrieved it).
- `Filter` AST nodes: `And`, `Or`, `Not`, `Eq`, `In`, `Range`, `Exists`, `Contains`.
- `QueryOp` is a discriminated union. Every op has `source`, an optional
  `collection`, an optional `filter`, and `limit`:
  - `Lexical(text, fields?)`
  - `Vector(content | hyde_text, field)`: the harness embeds via the bound embedder
  - `Hybrid(text, field, weights?)`
  - `FilterOnly()`
  - `Regex(pattern, fields?)`, which also covers glob for files
  - `Traverse(start: Filter, rel_types, direction, depth, target_label?)`
  - `Aggregate(group_by, metrics)`: returned as structured hits
  - `Fetch(doc_ids)`
  - `Native(dialect, query)`: gated, see §6
- `Capability` enum: `LEXICAL, VECTOR, HYBRID, FILTER, REGEX, TRAVERSE,
  AGGREGATE, FETCH, NATIVE, IMAGE_QUERY`.
- `Manifest` covers one source:
  - collections (tables, indices, labels, dirs)
  - fields: name, type, and flags for searchable, filterable, sortable and vector
  - vector fields: dimension, metric, and `embedder_id`
  - relationship types (graphs)
  - low-cardinality sample values and counts
  - capabilities
  - free-text `annotations` merged from an optional user file

### 3.3 Backend protocol (`backends/base.py`)

```python
class Backend(Protocol):
    name: str
    def capabilities(self) -> set[Capability]
    async def discover(self, detail: DiscoverDetail = "summary",
                       collection: str | None = None) -> Manifest
    async def execute(self, op: QueryOp) -> list[Hit]
    async def close(self) -> None
```

The base class enforces:

- a maximum `limit` per op (config, default 100)
- a per-call timeout
- `LIMIT` injection for SQL, Cypher and BigQuery
- read-only behavior

Adapters translate filters through `filters.py`.

Per-adapter notes:

| Adapter | Lexical | Vector | Other |
|---|---|---|---|
| files | rank_bm25 in memory | local embed index (optional) | Regex/glob over paths & contents, Fetch reads file; images by extension |
| opensearch | `match`/`multi_match` | `knn` | Hybrid via search pipeline or client-side RRF; Aggregate via aggs |
| pgvector | `tsvector` / `ts_rank` | `<=>` / `<->` | Filter via SQL; Aggregate; Native SQL |
| mysql | `MATCH … AGAINST` | vector type where available (MySQL 9 / HeatWave); else none | Filter, Aggregate, Native SQL |
| neo4j | full-text index | vector index | Traverse via Cypher; Native Cypher |
| milvus | BM25 sparse (Milvus 2.5+) if present | ANN | Filter via Milvus expr; Hybrid via `hybrid_search` |
| bigquery | `SEARCH()` | `VECTOR_SEARCH` | Filter, Aggregate, Native SQL; dry-run byte cap |

`discover()` introspects using each engine's own metadata: `_mapping`,
`information_schema`, `db.schema.*`, collection describe, and the BigQuery
`INFORMATION_SCHEMA`. It samples values for low-cardinality fields and caches
the result. An optional YAML annotations file adds meaning that introspection
cannot find (e.g. "`dx_code` is ICD-10-CM") and binds vector fields to embedder
ids when metadata doesn't say.

### 3.4 Embedders (`embedders/`)

```python
class Embedder(Protocol):
    id: str
    modalities: set[Modality]
    dim: int
    async def embed(self, items: list[Content],
                    purpose: Literal["query", "document"]) -> list[Vector]
```

Adapters:

- `vertex`: Gemini text embeddings with a task type per purpose, and the
  multimodal embedding model.
- `openai_compat`
- `tei`
- `http_generic`: configured with a URL, a request template for text and for
  base64 images, a response JSONPath, batch size and headers. Covers custom
  containers such as MedSigLIP on Azure ML/AKS.
- `local`: sentence-transformers and CLIP.

Auth providers are separate from adapters: `ApiKey`, `Bearer`, `GcpAdc`
(google-auth) and `AzureIdentity` (azure-identity, managed identity or CLI
credential).

Common behavior: batching, retries with backoff and jitter, and an optional
LRU cache for query embeddings.

**Embedding-space binding:** each vector field in a manifest carries
`embedder_id`. The executor resolves the embedder from a registry by that id.
If no compatible embedder is registered, or the modality isn't supported (e.g.
an image query against a text-only embedder), the executor returns a
`ToolError` to the planner. It never falls back to a different model silently.

### 3.5 Model roles (`models/base.py`)

```python
class Driver(Protocol):
    id: str
    supports_images: bool
    async def plan(self, state: PlannerView, tools: list[ToolSpec]) -> list[ToolCall]
    async def run_delegate(self, question: Query, tools: ToolRuntime,
                           budget: Budget) -> list[Hit]   # model-centric mode

class Decider(Protocol):
    id: str
    async def judge(self, question: Query, hits: list[Hit]) -> list[Judgment]
    async def decide(self, state: ControllerView) -> Decision
```

- `Judgment(doc_key, p_relevant: float, rationale: str | None)`
- `Decision(action: CONTINUE|REFINE|BROADEN|SWITCH_SOURCE|STOP, confidence, note)`
- A decider may implement only `judge` or only `decide`. The Analyzer and
  Controller roles each take their own decider, so e.g. a cross-encoder judges
  while Jev decides whether to stop.

Driver adapters:

- `anthropic`: Messages API with native tool use.
- `openai_compat`: chat completions with tools. Covers vLLM, Ollama, Together,
  Baseten-hosted SID-1 and OpenAI.

`run_delegate` has a default: a plain tool loop over the same `ToolRuntime`.
Models with a special protocol (e.g. SID-1's ranked-list output) can override
it.

Decider adapters:

- `typesafe`: System One models via the TypeSafe API. Relevance becomes a
  typed judgment with a probability; stop/continue becomes a typed decision.
- `cross_encoder`: local judge only.
- `llm_judge`: wraps any Driver model with a grading prompt and structured
  output.

## 4. Search loop (`core/harness.py`)

```python
harness = Harness(backends=[...], embedders=[...], driver=..., analyzer=...,
                  controller=..., mode="harness", budget=Budget(...), hooks=...)
result = await harness.search(question, sources=None, top_k=20)
# result: SearchResult(hits: list[RankedHit], trace: Trace, stop_reason, usage)
```

### 4.0 Setup (once per Harness)

1. `discover()` every backend.
2. Merge the annotations.
3. Cache the manifests.
4. Build the tool specs from capabilities.
5. Render a compact manifest summary for the planner.

### 4.1 Plan

The planner sees:

- the question
- the manifest summaries
- the digest of previous turns (queries tried, hit rates, judged-relevant and
  irrelevant examples)
- the controller's last directive

It returns a batch of tool calls, and may call `discover(source, detail="full",
collection=...)` for depth. The prompt encourages 4–8 diverse parallel calls per
turn: narrow and broad lexical searches, HyDE vector searches and filtered
searches, as SID-1 does.

### 4.2 Execute

1. Validate each call against the manifest: fields exist, the op is supported,
   an embedder is bound, and the native guard passes.
2. Resolve embeddings.
3. Run everything concurrently (`asyncio.gather`, with a per-call timeout and a
   cap on concurrency per source).
4. Merge the hits into the `CandidatePool`, keyed by `(source, doc_id)` and
   accumulating provenance.

Failures become `ToolError`s that the planner sees on its next turn.

### 4.3 Analyze

1. The analyzer decider judges **only new candidates**, in batches.
2. Fused score: reciprocal rank fusion across the ops that retrieved each doc,
   combined with the judge's `p_relevant` using a configurable blend (default:
   judge probability dominates, RRF breaks ties).
3. Build a compact **digest** for the planner: the top new hits with short
   snippets and verdicts, clearly irrelevant patterns, and per-query and
   per-source yield. Raw results never go back to the planner in bulk.

### 4.4 Decide

The controller decider returns a `Decision`. Hard budget limits override it:
maximum turns, tool calls, tokens, cost (USD) and wall-clock time.

The default heuristic controller is used when no decider is configured. It
stops when a turn adds no new relevant hits above a threshold, or when the
top-k is stable across two turns.

### 4.5 Finalize

1. Sort the pool by final score.
2. Truncate to `top_k`.
3. Attach evidence to each hit: the judge rationale, the provenance ops, and
   the scores.
4. Return the `SearchResult`, including `stop_reason` (`CONTROLLER_STOP`,
   `BUDGET_TURNS`, `BUDGET_COST`, `BUDGET_TIME`, …).

### 4.6 Modes (`core/modes.py`)

| Mode | Behavior |
|---|---|
| `retrieval` | One Plan+Execute turn, no judging; rank by fusion of backend scores (optionally rerank with a cross-encoder) |
| `harness` | Full loop §4.1–4.5 |
| `model` | `driver.run_delegate()` replaces §4.1–4.4; the harness still validates tool calls, enforces the budget, and records the trace; optional analyzer re-rank at finalize |

## 5. Trace

Every step produces a `TraceEvent` (plan, tool_call, tool_result summary,
tool_error, judgment batch, decision, budget tick) with timestamps, durations,
and token and cost usage. The trace is JSON-serializable for eval and
debugging. Secrets never enter the trace. Raw content can be excluded per
source (see §6).

## 6. Error handling and safety

- **Tool errors** are structured `ToolError`s returned to the planner. They are
  never raised.
- **Raised exceptions** are only harness-level: no backend reachable, the
  driver unavailable after retries, or an invalid config.
- **Retries** use exponential backoff with jitter for backend, embedder and
  model calls. Timeouts apply per call.
- **Decider failures** fail open: the hit keeps its fused score and is marked
  `unjudged`, and a trace event records it.
- **Partial results** are always returned when a budget stops the run.
- **Read-only enforcement:**
  - Adapters only issue reads.
  - `native_query` is disabled by default and enabled per source. When enabled:
    - SQL must parse (sqlglot) as a single `SELECT` with no DML/DDL, and a
      `LIMIT` is injected.
    - Cypher is rejected if it contains `CREATE|MERGE|DELETE|DETACH|SET|REMOVE|CALL
      dbms`, and a `LIMIT` is injected.
    - OpenSearch is limited to the `_search` body.
  - The docs recommend read-only credentials.
- **Cost caps:** BigQuery dry-run with `maximum_bytes_billed`; result and row
  caps on every op.
- **Hooks:** `on_before_model_call(payload) -> payload` and
  `on_trace_event(event)`. Every call to an external model or embedder goes
  through the first. The defaults are no-ops; the medical layer will implement
  redaction and audit here.
- **Source policy:** per-source `allowed_model_ids` restricts which drivers,
  deciders and embedders may receive that source's raw content (e.g. a PHI
  source limited to self-hosted models). Hits from a restricted source are sent
  only as ids and metadata to disallowed models.
- **Secrets** come from environment variables or cloud credentials. They are
  never read from config files and never written to traces.

## 7. Configuration

Harnesses can be built in code or from YAML:

```yaml
backends:
  - {name: notes, type: pgvector, dsn_env: NOTES_DSN, native_query: false}
  - {name: images, type: milvus, uri_env: MILVUS_URI}
embedders:
  - {id: "vertex:gemini-embedding-001", type: vertex, project_env: GCP_PROJECT}
  - id: "azure:medsiglip-448"
    type: http_generic
    url_env: MEDSIGLIP_URL
    auth: {type: azure_identity, scope: "api://medsiglip/.default"}
    request: {image: {"instances": [{"image_b64": "{{b64}}"}]}}
    response_path: "$.predictions[*].embedding"
    dim: 1152
annotations: annotations.yaml
driver: {type: anthropic, model: claude-sonnet-5-5}
analyzer: {type: cross_encoder, model: BAAI/bge-reranker-v2-m3}
controller: {type: typesafe, model: jev}
mode: harness
budget: {max_turns: 4, max_tool_calls: 32, max_cost_usd: 0.50, max_seconds: 60}
```

## 8. Testing

- **Unit tests:**
  - golden tests for Filter AST → each dialect
  - manifest validation of ops
  - native guard accept/reject cases
  - RRF and score blending
  - budget and stop logic
  - the full loop with scripted fake Driver/Decider/Embedder, so it is
    deterministic and makes no network calls
- **Adapter contract suite:** one parametrized suite that every backend must
  pass (discover shape, each advertised capability, filters, limit caps,
  read-only rejection).
  - Runs against docker-compose services (OpenSearch, Postgres+pgvector, MySQL,
    Neo4j, Milvus) loaded with a shared fixture corpus.
  - BigQuery runs against a real test dataset and is skipped without
    credentials.
- **Model and embedder adapters:** recorded HTTP responses (respx) for
  Anthropic, OpenAI-compatible, TypeSafe, Vertex, TEI and generic HTTP.
  Optional live smoke tests are marked `live`.
- **Minimal eval:** a BEIR loader, NDCG@10 / Recall@100 / MRR, and cost and
  latency per run. Ships with NFCorpus (medical-flavored) to compare the three
  modes on the files and pgvector backends.

## 9. Success criteria

1. All seven v1 backends pass the contract suite.
2. The same `Harness.search()` call runs unchanged across modes, drivers and
   deciders by changing only config.
3. On NFCorpus, `harness` mode with an LLM or cross-encoder judge beats
   `retrieval` mode on NDCG@10 over the same backend, with cost and latency
   reported.
4. A vector op against a field whose embedder isn't registered yields a
   `ToolError`, never results from a different embedding space.
5. `native_query` rejects every write/DDL case in the guard test corpus.
