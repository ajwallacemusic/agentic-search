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
