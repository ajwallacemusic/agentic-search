# Search Service (HTTP + SSE) — Design Spec

- **Date:** 2026-09-29
- **Status:** Approved 2026-09-29 (combined design for streaming sub-projects 2 and 3)
- **Scope:** Streaming sub-project 2 of 3. It builds on spec 1
  (`2026-09-29-streaming-events-design.md`, event schema v1). Spec 3 (TypeScript client) consumes
  this service and its exported contract.

## 1. Purpose

Other projects, including non-Python ones, can run agentic searches over HTTP and watch them
live. An operator configures datastores and models once as named **profiles**. Clients choose a
profile and override per-request options (sources, mode, budget, result size), but never send
credentials or datastore definitions.

### Goals

1. `agentic-search serve --config service.yaml` runs a FastAPI app under uvicorn.
2. `POST /v1/search/stream` streams spec-1 events as server-sent events. `POST /v1/search` returns
   the result as JSON.
3. Per-profile limits cap what a request may ask for (budget ceilings, content access).
4. Closing the connection cancels the search, including in-flight backend and model calls.
5. The wire contract is exported as the event JSON Schema plus golden SSE fixtures, committed to
   the repo, and a test keeps both in step with the models.
6. Text and image questions work. Images are always inline base64.

### Non-goals

- Multi-tenancy beyond API keys, per-key quotas, rate limiting over time (only a concurrency cap),
  OAuth/OIDC.
- Clients registering datastores or models at runtime.
- Persisting searches or resuming a stream (`Last-Event-ID` is ignored).
- The TypeScript client (spec 3).

## 2. Prerequisite fix: image bytes in JSON

`ImagePart.data` (bytes) could not be JSON-serialised when not valid UTF-8, so any event
carrying an image question or image content failed to serialise.

`ImagePart` now serialises `data` as **standard** base64 (`+`/`/`, so browser `atob` works)
through a JSON-mode field serializer. It validates JSON input as base64, accepting both the
standard and URL-safe alphabets (`val_json_bytes="base64"`). Python-mode dumps keep `bytes`.

## 3. Packaging

- Optional extra `server = ["fastapi>=0.115", "uvicorn>=0.30"]`. Both are also added to the dev
  group.
- Console script `agentic-search = "agentic_search.server.cli:main"`.
- Package `agentic_search.server`:

| Module | Responsibility |
|---|---|
| `config.py` | `AuthConfig`, `ServiceConfig`, `ProfileLimits`, `Profile`, `load_service(path)`, `check_profiles` |
| `models.py` | Request models (`SearchRequest`, `StreamRequest`, `ImageInput`, `BudgetOverride`), `build_query`, `resolve_budget`, `lean_result`, `render_event`, `RequestError` |
| `sse.py` | `format_sse`, `KEEPALIVE`, `sse_body` (framing, keep-alives, disconnect → cancel) |
| `app.py` | `create_app(config, profiles)`: routes, auth, capacity gate, lifespan |
| `demo.py` | `demo_harness`, `demo_app`: an in-memory service needing no network or models, used for fixtures and client end-to-end tests |
| `schema.py` | `event_schema`, `fixture_streams`, `render_files`, `export` |
| `cli.py` | `serve` and `export-schema` subcommands |

## 4. Configuration

```yaml
service:
  auth: {type: api_key, keys_env: SEARCH_API_KEYS}   # or {type: none} — required, explicit
  cors_origins: ["https://app.example.com"]          # default []: no CORS middleware
  max_concurrent_searches: 16                         # default 16, ≥1
  default_profile: general                            # optional
  keepalive_s: 15                                     # default 15, >0
  max_image_bytes: 10000000                           # default 10 MB per image
  max_images: 4                                       # default 4 per request
profiles:
  general: {<harness config as build_harness accepts>}
  strict:
    <harness config>
    limits: {max_budget: {max_turns: 2, max_cost_usd: 0.25}, allow_include_content: false}
```

- `service.auth` is required. `api_key` needs `keys_env`, a comma-separated list read at app
  creation. An empty or unset variable is a `ConfigError`. Every key is registered for scrubbing.
- Unknown keys in `service:`, `auth:` and `limits:` are errors (`extra="forbid"`).
- `limits.max_budget` maps `Budget` field names to positive ceilings. Unknown field names, or
  ceilings that are not a valid `Budget` value, are errors. `allow_include_content` defaults to
  true.
- `load_service` builds every profile's harness with `build_harness(cfg, base_dir=<yaml dir>)`.
  It does not set them up. Errors are raised as `ConfigError`.
- `create_app(config, profiles)` accepts `Profile` objects or bare `Harness` objects (default
  limits). It requires at least one profile, and `default_profile` must name one.

## 5. HTTP API

All `/v1/*` routes require auth when `auth.type == api_key`. The key is sent as
`Authorization: Bearer <key>` or `X-API-Key: <key>` and compared in constant time. A failure
returns `401` with `WWW-Authenticate: Bearer`.

| Route | Response |
|---|---|
| `GET /healthz` (no auth) | `{"status": "ok" \| "degraded", "version"}`; `ok` when every profile has discovered manifests |
| `GET /v1/profiles` | `{"profiles": [{name, default, available, error, mode, budget, limits, sources: [{name, backend_type, capabilities, collections, description}], setup_errors}]}` |
| `POST /v1/search` | lean result JSON (§6) |
| `POST /v1/search/stream` | `text/event-stream` (§7) |

### 5.1 Request body

`SearchRequest` (extra fields forbidden):
- `profile?`
- `question` (1–10 000 characters)
- `images?: [{data, mime}]`, where `mime` matches `^image/…`
- `sources?`
- `mode?: retrieval|harness|model`
- `top_k` (1–1000, default 20)
- `budget?`: any subset of `Budget` fields, each positive
- `include_content` (default false)
- `include_trace` (default false)

`StreamRequest` adds `snapshot_k` (0–100, default 10). A body that fails validation returns
`422`.

- **Profile resolution:** the request's `profile`, else `default_profile`, else the only profile.
  If none of these resolves, the response is `400`. An unknown profile is `404`.
- **Images** are decoded from standard or URL-safe base64 into `ImagePart(data=…)`:
  - Invalid base64, more than `max_images` images, or an image over `max_image_bytes` returns
    `400`.
  - A `uri` field is rejected by validation with `422`. The service never reads a
    client-supplied path or URI.
- **Budget:** the profile harness's budget, updated with the request's fields, then capped by
  `max_budget`. An unlimited (`None`) field under a ceiling becomes the ceiling.
- **Content:** `include_content` on a profile with `allow_include_content: false` returns `400`.

### 5.2 Readiness and errors

- Profiles are set up in the app lifespan and lazily on first use, so the app also works without
  lifespan events, for example in tests with `httpx.ASGITransport`. They are closed at shutdown.
- A profile whose setup raises `HarnessError` (no backend could be discovered) returns `503` on
  search. `/v1/profiles` lists it with `available: false` and a scrubbed `error`.
- In `/v1/search`, a `HarnessError`, for example unknown sources, returns `400` with the scrubbed
  message. Any other exception returns a generic `500`, with no detail leaked.
- **Capacity:** there is a non-blocking gate of `max_concurrent_searches`. A search that cannot
  get a slot immediately returns `429`. The slot is held for the whole search: for streams, it is
  released when the response ends in any way, even if the body never starts.

## 6. Lean result

The JSON view of a `SearchResult` is `result.model_dump(mode="json")`, with two changes:
- `trace` is removed unless `include_trace`.
- Each `hits[i].hit.content` key is removed unless `include_content`.

The `/v1/search` body is exactly this lean result. The streamed `search_finished` event carries
it as `result`. Every other event is its spec-1 JSON unchanged. `HitSummary.content` in
snapshots is present only if the request set `include_content`, which is passed through to
`Harness.stream`.

## 7. SSE stream

- Response headers: `Content-Type: text/event-stream`, `Cache-Control: no-cache`,
  `X-Accel-Buffering: no`, `X-Search-Profile: <name>`.
- Each event is framed as:
  ```
  id: <seq>
  event: <type>
  data: <single-line event JSON>

  ```
- After `keepalive_s` seconds without an event, a `: keep-alive` comment frame is sent. Waiting
  for the keep-alive never cancels the pending event.
- Validation errors (`4xx`) happen before streaming starts. Once streaming, a search failure is a
  `search_failed` event, and the stream ends after the terminal event.
- **Client disconnect:** the body watches the ASGI `receive` channel for `http.disconnect`. When
  it arrives, the body stops, cancels the pending event, and closes the `SearchStream`, which
  cancels the search task and its in-flight calls (spec 1 §4.4). This does not depend on the next
  write failing, so cancellation is prompt even between keep-alives.

## 8. Contract export

- `event_schema()`: `TypeAdapter(SearchEvent).json_schema(mode="serialization")` with `title` and
  `$schema` (draft 2020-12), adjusted to describe the lean wire output (final review):
  `trace` is not in `$defs.SearchResult.required` and is described as "present only when the
  request sets include_trace"; `$defs.Hit.content` is described as "present only when the
  request sets include_content". A test validates every committed fixture frame's `data`
  against the committed schema (`jsonschema.Draft202012Validator`; `jsonschema` is a dev
  dependency).
- `fixture_streams()`: runs `demo_app()` in-process with four stream requests and records each
  one's SSE frames as `{id, event, data}`:
  - `retrieval` (`snapshot_k` 3)
  - `harness` (`snapshot_k` 2)
  - `with_content` (retrieval with `include_content` and `include_trace`)
  - `failed` (unknown source)

  Run-dependent values are normalised (`search_id` → `"fixture"`, `at_ms`/`duration_ms` → 0.0),
  and every float is rounded to 6 decimals, so the output is byte-stable across runs,
  numpy/BLAS builds and architectures.
- `agentic-search export-schema --out schema/` writes `search-events.v1.schema.json` and
  `search-events.v1.fixtures.json` (sorted keys, 2-space indent). Both are committed under the
  repo's `schema/`. A test fails if they differ from freshly rendered output.

## 9. Demo service

`demo_harness(slow_s=None, on_cancel=None)` builds a harness from:
- four documents (d1–d4, with titles) on a `FilesBackend`;
- `EchoDriver("docs", limit=10)`;
- `KeywordJudge(["headache"])`.

With `slow_s`, every backend call first sleeps and calls `on_cancel` if cancelled.
`demo_app(**extra_profiles)` serves the `demo` profile plus any extra profiles, with auth off.

## 10. Testing

Everything runs in-process with no services, using `httpx.ASGITransport` and a real uvicorn on a
free port:
- Lean results and the `include_content`/`include_trace` flags; the profile content ban.
- Profile resolution and the default-profile check.
- API-key auth (bearer, `X-API-Key`, wrong, missing), an unauthenticated `healthz`, and a missing
  keys variable.
- `/v1/profiles` contents; an unavailable profile gives `503`, and health reports `degraded`.
- Budget clamping end to end.
- Images: decoded inline, invalid base64 `400`, `uri` `422`, too many `400`.
- `HarnessError` gives `400`, and capacity gives `429` for both routes.
- Stream framing: ids equal `seq`, event equals `type`, the lean finish, `snapshot_k`, and
  `search_failed` as an event.
- The slot is released after streams.
- `format_sse`, keep-alives during silence without dropping events, disconnect via `receive`
  cancelling the backend call, and closing the body cancelling it.
- A live uvicorn stream and search; a live client disconnect cancels the backend call within
  seconds with `keepalive_s=30`.
- The schema-drift test, the event-type coverage of the schema, `load_service` (valid and invalid),
  `ProfileLimits` validation, `resolve_budget`, and the CLI `export-schema`.
- An `ImagePart` JSON round trip with standard base64, accepting URL-safe input.
