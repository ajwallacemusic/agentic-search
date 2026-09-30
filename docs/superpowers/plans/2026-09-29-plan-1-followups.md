# Plan 1 follow-ups (carry into Plans 2 and 3)

Recorded at the end of Plan 1 execution (branch feat/plan-1-core). Items marked **Required** must land before
the named plan ships the feature that exposes them.

## Required
- **Plan 2 — secret scrubbing.** Backend exception text flows unscrubbed into `ToolError.message`, trace events and
  `Harness.setup_errors`. Before any DSN/URL-bearing adapter (pgvector, MySQL, BigQuery, OpenSearch) merges: have
  `config.resolve_env` record resolved secret values, and scrub them plus URL userinfo in `Trace.add` and ToolError construction.
- **Plan 2 — delegate model spend after budget.** `ToolCallingDriver.run_delegate` keeps calling the model (up to
  `max_turns`) after `_DelegateRuntime` reports `BUDGET_COST`/`BUDGET_TOKENS`; tools stop but model spend continues.
  Expose `runtime.exhausted` (or a flag on outputs) and force `finish` immediately.
- **Plan 3 — hooked backend embeddings.** Backends embed documents directly (`FilesBackend._build_vectors`), bypassing
  `Hooks.before_model_call` and `SourcePolicy`. Wrap every embedder handed to a backend (e.g. `HookedEmbedder`) before any
  remote embedder (Vertex, TEI, generic HTTP/MedSigLIP) is usable from a backend.

## Known, non-blocking
- `_DelegateRuntime.call` does not validate the length/type of hook-returned outputs (a buggy hook aborts a model-mode search).
- A hook raising `NotImplementedError` latches the analyzer/controller decider off.
- `on_trace_event` listener exceptions propagate into the search.
- Spec §5 per-event token/cost usage on plan/judge/decision trace events is not recorded (totals only).
- Judge batches run sequentially.
- FilesBackend: `discover` ignores `detail`/`collection`; `infer_field_type` samples 50 values; regex matches `""` for image-only docs;
  `Hit.metadata` shares the index dict.
- Config: missing keys raise raw `KeyError`; duplicate embedder ids overwrite.
- Clients: empty `choices` → IndexError (OpenAI-compat); empty assistant message → `content: []` (Anthropic).
- Various test-coverage gaps noted in task reviews (filters edge cases, executor branches, analyzer timeout/partial batch).

## Status after Plan 2 (2026-09-29)
- Done in Plan 2: secret scrubbing; delegate model-spend cutoff.
- Still required for Plan 3: hooked backend embeddings.

## Plan 2 follow-ups (carry into Plan 3)
- Postgres lexical search recomputes `to_tsvector(...)` per row (no index use); support a stored/discovered tsvector column.
- BigQuery verified only against a fake client; run `test_live_bigquery` against a real dataset.
- BigQuery composite primary keys (table_constraints) still use the first column.
- `ValueError` from backend constructor validation (BigQuery project/dataset, Postgres text_search_config) is not converted to `ConfigError`.
- `strip_collection` cannot distinguish a raw pk that itself starts with "<collection>/" (namespaced form works).
- Minor: remaining deferred items are listed in the Plan 2 ledger summary (final message of the run).

## Status after Plan 3
- Done: hooked backend embeddings (Task 2); Postgres stored tsvector and ConfigError for invalid values (Task 8).
- Open: live runs of BigQuery, Vertex, OpenAI/TEI embeddings, TypeSafe and Azure/GCP auth with real credentials; Neo4j Enterprise/Aura/TLS; auth-enabled Milvus.

## Plan 3 final review — deferred
- Hook payload shape for backend document embeddings (`HookedEmbedder` → `before_model_call`).
- Default embedder ids.
- Postgres tsvector logging.
- `CachedEmbedder`-over-`HookedEmbedder` rebinding.
- httpx `AsyncClient` event-loop binding (embedders, TypeSafe).
- Neo4j regex wrapper can be escaped by unbalanced parentheses (semantics only; the pattern is a `$param`).
- Milvus `_samples` reads only a 1000-row slice (`SAMPLE_SCAN_MAX`) for distinct sample values.
- `OpenAICompatClient`/`AnthropicClient` expose no `close()`, so `Harness.close` cannot release their SDK clients.
- `AzureIdentity.close` closes a caller-supplied credential and never resets it, so an Azure-auth embedder shared by two harnesses breaks after the first closes.
- `TypeSafeDecider.judge` drops usage accumulated from earlier batches when it re-raises a later failure.
- `Harness.close` discards close exceptions (`return_exceptions=True`) without logging them.
- Milvus caches a positive load state forever; a collection released later gives a server error instead of "not loaded".
- The Cypher guard and wrapper rely on `CALL { WITH x … }`, which Neo4j flags as deprecated in favour of `CALL (x) { … }`; revisit when it is removed.
- Native Cypher rejects any backslash-u sequence (Neo4j decodes them before tokenising); queries needing non-ASCII literals must use literal characters or parameters.

## Plan 4 final review — deferred

- Generic BaseModel-aware scrubbing in `scrub_data`; today it does not descend into BaseModel payloads, so event builders must scrub such fields by hand (e.g. `PhaseSummary.note`).
- SSE payload size of `SearchFinished` (full trace plus content): the service layer should offer a lean projection.
- Snapshot ranking and the turn summary each sort the pool; share one sort.
- `NullEmitter` still constructs event objects before discarding them.
- Stream tests' phase pairing check compares ordered `(phase, turn)` lists but not start-before-finish ordering, and would reject nested phases if any are added.
- The `SearchStarted.mode` drift test compares against a hard-coded set rather than `harness._MODES`.
- The delegate trace entry records the post-run turn while `phase_finished(delegate)` uses the start turn.

## Plan 5 final review — deferred

- CORS headers on `500` responses: `ServerErrorMiddleware` sits outside `CORSMiddleware`, so a browser client sees a generic `500` as a CORS failure.
- A runtime `HarnessError` during `/v1/search` maps to `400`; some are not the client's fault (e.g. every backend failing mid-search) and might deserve a `5xx`.
- Profile names that are not latin-1 cannot be sent back in the `X-Search-Profile` header.
- Startup blocks for up to each backend's `discover_timeout` while the lifespan sets profiles up; document it (or set up in the background).
- Per-key quotas and rate limiting over time (only a concurrency cap exists).
- `/v1/profiles` reports `limits` as configured, not the effective budget ceilings (which now default to the profile budget).
- `/v1/search` disconnect race: `suppress(CancelledError)` around awaiting the cancelled search can swallow a cancellation of the handler itself (finally still stops the search and releases the slot).
- `agentic-search serve` still prints a traceback (not a one-line error, exit 2) for YAML syntax errors, a missing config file, and malformed harness configs (e.g. `backends: 5`).
- Per-profile setup locks are created in `create_app`; reusing one app across event loops while a lock is contended raises.

## Plan 6 final review — deferred

- Full value-type drift checking between the TS types and the schema (for example generating the TS types from the JSON Schema); the contract tests check field sets, nullability, enums and `$def` coverage only.
- `parseSse` strips a leading BOM although `TextDecoder("utf-8")` already does by default; the strip is redundant.
- `parseSse` re-scans and re-slices the buffer per line, so a very long line arriving in many chunks is parsed in quadratic time.
- `test/e2e.test.ts` listens for the child's `error` event only inside `waitFor`; after startup no listener remains, so a later child `error` (for example from killing it) would surface as an unhandled error. Add a permanent listener.
- TS client: `NullableKeys` counts `unknown`/`any` fields as nullable while pydantic `Any` fields have no null branch — adding such a field fails the contract test confusingly (fails loudly, not silently).
- TS client: `ClientOptions.fetch` doc mentions "older runtimes", but the client now also needs a global `Headers`; document or accept a `Headers` polyfill.
- TS client: very long JSON `detail` strings / 422 arrays go into `AgenticSearchError.message` untruncated.
- TS client e2e checks only top-level keys of `ProfileInfo.limits`/`budget`.
