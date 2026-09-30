# TypeScript Client — Design Spec

- **Date:** 2026-09-29
- **Status:** Approved 2026-09-29 (combined design for streaming sub-projects 2 and 3)
- **Scope:** Streaming sub-project 3 of 3. It consumes the service of spec 2
  (`2026-09-29-search-service-design.md`) and event schema v1 (spec 1).

## 1. Purpose

A web app or a Node, Deno or Bun project can run searches against the agentic-search service
with typed results and live, typed progress events.

### Goals

1. No runtime dependencies. It uses the standard `fetch`, `Headers`, `ReadableStream` and
   `TextDecoder` APIs, so it works in browsers, Node ≥ 18, Deno and Bun.
2. `stream()` is an async iterator of a discriminated `SearchEvent` union that mirrors schema v1.
3. Leaving the loop, or aborting an `AbortSignal`, closes the connection, and the service then
   cancels the search.
4. Drift between the TypeScript types and the Python models in field sets, nullability,
   enumerations or `$def` coverage fails the test suite, both at compile time and at run time.
5. An end-to-end test runs against the real Python service.

### Non-goals

- Reconnection or resume.
- Browser `EventSource`: it cannot POST a body or send an auth header.
- Publishing to npm. The package is `private: true` until the owner decides to publish.

## 2. Package

`clients/typescript/` is an npm package named `@agentic-search/client`, version 0.1.0:
- ESM only (`"type": "module"`), with `exports["."] = {types, import}` pointing into `dist/`.
- `engines.node >= 18`.
- Dev dependencies: `typescript` ^7, `vitest` ^5, `@types/node` ^22.
- Scripts: `build` (`tsc -p tsconfig.build.json`), `typecheck` (`tsc --noEmit`, which covers the
  tests too), and `test` (`vitest run`).
- `tsconfig`: `target ES2022`, `module`/`moduleResolution` `NodeNext` (relative imports carry
  `.js`), `lib ES2022 + DOM + DOM.Iterable`, `strict`, `noUncheckedIndexedAccess`, and
  declarations.
- `.gitignore`: `node_modules/`, `dist/`. `package-lock.json` is committed.

| Module | Responsibility |
|---|---|
| `src/events.ts` | Schema-v1 types, `SCHEMA_VERSION`, `EVENT_TYPES`, `isSearchEvent`, `isTerminal`, `hitKey` |
| `src/errors.ts` | `AgenticSearchError` (`status`, `detail`) |
| `src/sse.ts` | `parseSse(body)`: a spec-following SSE parser |
| `src/client.ts` | `AgenticSearchClient`, request/response types |
| `src/index.ts` | Public exports |

## 3. Types (`events.ts`)

The TypeScript side mirrors the Python models field for field:
- **Enumerations:** `Phase`, `Mode`, `Action`, `ToolErrorKind` and `StopReason` are literal unions.
- **Content:** `TextPart`, `ImagePart` (`data` is standard base64 or null, `uri` is string or
  null), `StructuredPart`, `ContentPart`, `Query`.
- **Supporting models:** `Budget`, `Usage`, `PhaseSummary`, `ToolErrorInfo`, `HitSummary`,
  `OpRef`, `Hit`, `RankedHit`, `TraceEvent`, `Trace`, `SearchResult`.
- **Lean-result differences:**
  - `Hit.content?` is optional, because the service omits it unless `include_content`.
  - `SearchResult.trace?` is optional, because the service omits it unless `include_trace`.
- **Events:** `EventBase` carries `schema_version`, `search_id`, `seq`, `turn` and `at_ms`. There
  is one interface per event type, and `SearchEvent` is their union on `type`.
- `EVENT_TYPES` lists the nine types. `isSearchEvent(value)` checks for a known `type`.

## 4. SSE parser (`sse.ts`)

`parseSse(body: ReadableStream<Uint8Array>)` yields `{event, data, id}`, following WHATWG
HTML §9.2.6:
- It decodes UTF-8 in streaming mode, so multi-byte characters split across chunks are safe.
- It accepts `\n`, `\r\n` and `\r` line endings. A trailing `\r` waits for the next chunk.
- It strips a leading BOM.
- It ignores comment lines (`:`).
- Field parsing: `field: value` has one optional leading space removed. A field with no colon has
  an empty value. `data` lines are joined with `\n`. `event` defaults to `message`. An `id`
  containing NUL is ignored.
- A message is dispatched on a blank line, but only if it had `data`. An unterminated message at
  end of stream is discarded.
- If the consumer stops early or throws, the reader is cancelled, which closes the body.

## 5. Client (`client.ts`)

`new AgenticSearchClient({baseUrl, apiKey?, fetch?, headers?})`:
- A trailing slash on `baseUrl` is trimmed.
- `apiKey` is sent as `Authorization: Bearer`.
- `headers` are added to every request, merged case-insensitively through a `Headers` object; the
  client's own `Accept`, `Content-Type` (when there is a body) and `Authorization` (when `apiKey`
  is set) replace any user header of the same name.
- `fetch` defaults to the global one, and a `TypeError` is thrown if none exists.

| Method | Request |
|---|---|
| `health({signal}?)` | `GET /healthz` → `{status, version}` |
| `profiles({signal}?)` | `GET /v1/profiles` → `ProfileInfo[]` |
| `search(request, {signal}?)` | `POST /v1/search` → lean `SearchResult` |
| `stream(request, {signal, onUnknownEvent}?)` | `POST /v1/search/stream` with `Accept: text/event-stream` → `AsyncGenerator<SearchEvent>` |

- **Request types.** `SearchRequest` and `StreamRequest` mirror spec 2 §5.1, with `images: {data,
  mime?}[]` and `budget: Partial<Budget>`. JSON bodies are sent with
  `Content-Type: application/json`.
- **HTTP errors.** A non-2xx response rejects with `AgenticSearchError(message, status, detail)`.
  `detail` is the body's `detail` field when the body is JSON, otherwise the parsed body or the
  raw text, always in full. The message is `<METHOD> <path> failed with <status>: <summary>`,
  where the summary is the `detail` string; or, for a FastAPI 422 `detail` array, each item as
  `loc.joined.by.dots: msg` (just `msg` without `loc`) joined by `; `; otherwise the status text
  (or `HTTP <status>` when it is empty), followed for a non-empty non-JSON body by ` — ` and the
  body with whitespace collapsed, truncated to 200 characters.
- **Stream semantics:**
  - Each SSE message's `data` is parsed as JSON. Known event types are yielded. Unknown types are
    passed to `onUnknownEvent(type, data)` and skipped.
  - The generator returns after the terminal event, `search_finished` or `search_failed`.
    `search_failed` is yielded like any other event, never thrown.
  - If the body ends before a terminal event and the stream was not aborted, it throws
    `AgenticSearchError(…, status 0)`.
  - The client owns an `AbortController` linked to the caller's `signal`. It aborts the
    controller in `finally`, so leaving the loop early closes the connection.
  - When the caller aborts, the pending read rejects with the runtime's `AbortError`.

## 6. Contract testing

The test suite reads `../../schema/search-events.v1.{schema,fixtures}.json`, which is exported by
the Python service (spec 2 §8).

- **Field lists, checked twice.** Per event type and per nested model there is a field list
  typed with an `Exactly<T, L>` helper. At compile time the list must name exactly the interface's
  keys; a missing or extra key is a type error. At run time the list must equal the schema's
  `properties` for that event type or model.
- **Nullability, checked twice.** Per event type and per nested model there is a list of the
  fields whose TS type admits `null`, typed with a `NullableKeys<T>` helper (keys `K` with
  `null extends T[K]`). At compile time the list must be exactly that set. At run time it must
  equal the schema properties that are nullable (an `anyOf` branch `{"type": "null"}`, or a
  `type` list containing `"null"`).
- **Enumerations.** `EVENT_TYPES` must equal the schema's discriminator mapping keys.
  `StopReason`, `Phase` (on both `PhaseStarted` and `PhaseFinished`), `Mode`, `Action` and
  `ToolErrorKind` must match the schema enums.
- **`$def` coverage.** Every schema `$def` is an event (a discriminator mapping target),
  `StopReason`, or a nested model with a field list, so a new nested model cannot go unchecked.
- **Not checked:** value types beyond nullability (e.g. `string` vs `number`, array item types);
  the fixtures and the end-to-end tests exercise those.
- **Fixtures.** Every frame has a known type, `event == data.type`, `id == data.seq`, and keys
  exactly equal to that event's field list. Each stream has exactly one terminal event, and it is
  the last. The lean result has no trace and no hit content. The `with_content` fixture has a
  trace, hit content and snapshot content.

## 7. End-to-end testing

`test/serve_demo.py` runs `demo_app(slow=demo_harness(slow_s=30, on_cancel=…))` under uvicorn on
a free port. It prints `LISTENING <port>`, and `CANCELLED` whenever a `slow` backend call is
cancelled. `test/e2e.test.ts` starts it with `uv run python serve_demo.py`, runs its tests, and
kills it afterwards. It checks:
- health and profiles;
- a streamed search ends in `search_finished` with gap-free `seq`, hits d1 and d4, and no trace;
- one-shot search returns the same hits;
- an unknown source gives a `search_failed` event;
- an unknown profile gives `404`, and an empty question gives `AgenticSearchError`;
- breaking out after `tool_call_started`, and aborting via `signal`, each make the server print
  `CANCELLED` within 10 s.

## 8. Docs

- `clients/typescript/README.md` covers usage, semantics and development.
- The root README's Service section links to it.
