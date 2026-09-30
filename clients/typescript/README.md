# @agentic-search/client

TypeScript client for the [agentic-search](../../README.md) service: one-shot search and
streamed search with typed events. No runtime dependencies — it uses the standard `fetch` and
`ReadableStream`, so it runs in browsers, Node ≥ 18, Deno and Bun.

```ts
import { AgenticSearchClient } from "@agentic-search/client";

const client = new AgenticSearchClient({ baseUrl: "https://search.example.com", apiKey: "…" });

for await (const ev of client.stream({ profile: "general", question: "what treats headache?" })) {
  switch (ev.type) {
    case "phase_started":
      console.log("…", ev.phase);            // plan | query | judge | decide | delegate
      break;
    case "results_updated":
      console.log(ev.hits.map((h) => h.title ?? h.key));
      break;
    case "search_finished":
      console.log(ev.result.hits.length, "hits", ev.stop_reason);
      break;
    case "search_failed":
      console.error(ev.error_type, ev.message);
      break;
  }
}
```

- `stream(request, { signal, onUnknownEvent })` yields `SearchEvent`s (a discriminated union on
  `type`, schema v1). The last event is `search_finished` or `search_failed`; a failed search is an
  event, not an exception.
- Breaking out of the loop or aborting `signal` closes the connection, and the service cancels
  the search. An aborted stream rejects with the runtime's `AbortError`.
- Events this client does not know (from a newer server: an unknown `type`, or a
  `schema_version` other than `SCHEMA_VERSION`) are skipped and reported to
  `onUnknownEvent(type, data)`.
- `search(request)` returns the lean result: no `trace` unless `include_trace`, no hit `content`
  unless `include_content`.
- `profiles()` lists the service's profiles, sources and limits; `health()` is the liveness check.
- HTTP failures reject with `AgenticSearchError` (`status`, `detail`): 400 a request the service
  can't run (for example `include_content` on a profile that forbids it, bad images, no profile
  when there are several, or unknown `sources` on one-shot search), 401 bad key, 404 unknown
  profile, 413 request body too large, 422 invalid request, 429 too many concurrent searches,
  503 profile unavailable.
- `results_updated` hits carry a snippet; full `content` comes only with `include_content`, if the
  profile allows it.
- Images go inline: `images: [{ data: <base64>, mime: "image/png" }]`.

## Develop

```bash
npm install
npm run typecheck && npm test   # e2e tests start the Python demo service with `uv run`
npm run build                    # emits dist/
```

`test/contract.test.ts` checks these types against the JSON Schema and golden SSE fixtures the
Python service exports to the repo's `schema/` directory (`agentic-search export-schema`): each
event's and nested model's field set, which fields are nullable, the enumerations, and that every
schema `$def` is covered, both at compile time and at run time. It does not check full value
types (for example `string` vs `number`); the golden fixtures and e2e tests exercise those.
