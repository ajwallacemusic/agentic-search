import { describe, expect, it, vi } from "vitest";
import { AgenticSearchClient, AgenticSearchError, type SearchEvent } from "../src/index.js";
import { streamOf } from "./helpers.js";

function frame(seq: number, data: Record<string, unknown>): string {
  return `id: ${seq}\nevent: ${String(data.type)}\ndata: ${JSON.stringify(data)}\n\n`;
}

const base = { schema_version: 1, search_id: "s", turn: 0, at_ms: 0 };
const started = { ...base, seq: 0, type: "search_started", question: { content: [] }, mode: "harness",
  sources: ["docs"], budget: { max_turns: 4, max_tool_calls: 32, max_tokens: null, max_cost_usd: null, max_seconds: 60 },
  setup_errors: {} };
const phase = { ...base, seq: 1, type: "phase_started", phase: "plan" };
const finished = { ...base, seq: 2, type: "search_finished", stop_reason: "no_plan",
  result: { question: { content: [] }, hits: [], stop_reason: "no_plan", mode: "harness",
    usage: { input_tokens: 0, output_tokens: 0, cost_usd: 0, tool_calls: 0, turns: 0 } } };

function sseResponse(chunks: string[]): Response {
  return new Response(streamOf(chunks), { status: 200, headers: { "Content-Type": "text/event-stream" } });
}

function fakeFetch(respond: (url: string, init: RequestInit) => Response) {
  return vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => respond(String(input), init ?? {}));
}

describe("AgenticSearchClient", () => {
  it("streams typed events until the terminal one, with auth and JSON body", async () => {
    const fetch = fakeFetch(() => sseResponse([": keep-alive\n\n", frame(0, started), frame(1, phase), frame(2, finished)]));
    const client = new AgenticSearchClient({ baseUrl: "http://svc/", apiKey: "k1", fetch });
    const seen: SearchEvent[] = [];
    for await (const ev of client.stream({ question: "q", snapshot_k: 3 })) seen.push(ev);
    expect(seen.map((e) => e.type)).toEqual(["search_started", "phase_started", "search_finished"]);
    const [url, init] = fetch.mock.calls[0]!;
    expect(url).toBe("http://svc/v1/search/stream");
    const headers = init!.headers as Record<string, string>;
    expect(headers["Authorization"]).toBe("Bearer k1");
    expect(headers["Accept"]).toBe("text/event-stream");
    expect(JSON.parse(init!.body as string)).toEqual({ question: "q", snapshot_k: 3 });
    const last = seen[2]!;
    if (last.type !== "search_finished") throw new Error("narrowing");
    expect(last.result.stop_reason).toBe("no_plan");
  });

  it("skips unknown event types and reports them", async () => {
    const future = { ...base, seq: 1, type: "thinking", text: "hmm" };
    const fetch = fakeFetch(() => sseResponse([frame(0, started), frame(1, future), frame(2, finished)]));
    const unknown: string[] = [];
    const client = new AgenticSearchClient({ baseUrl: "http://svc", fetch });
    const types: string[] = [];
    for await (const ev of client.stream({ question: "q" }, { onUnknownEvent: (t) => unknown.push(t) })) {
      types.push(ev.type);
    }
    expect(types).toEqual(["search_started", "search_finished"]);
    expect(unknown).toEqual(["thinking"]);
  });

  it("throws AgenticSearchError with the service detail on HTTP errors", async () => {
    const fetch = fakeFetch(() => new Response(JSON.stringify({ detail: "unknown profile 'x'" }), { status: 404 }));
    const client = new AgenticSearchClient({ baseUrl: "http://svc", fetch });
    await expect(client.search({ question: "q", profile: "x" })).rejects.toMatchObject({
      name: "AgenticSearchError", status: 404, detail: "unknown profile 'x'" });
    const stream = client.stream({ question: "q" });
    await expect(stream.next()).rejects.toBeInstanceOf(AgenticSearchError);
  });

  it("throws when the stream ends without a terminal event", async () => {
    const fetch = fakeFetch(() => sseResponse([frame(0, started)]));
    const client = new AgenticSearchClient({ baseUrl: "http://svc", fetch });
    const types: string[] = [];
    await expect((async () => { for await (const ev of client.stream({ question: "q" })) types.push(ev.type); })())
      .rejects.toMatchObject({ status: 0 });
    expect(types).toEqual(["search_started"]);
  });

  it("aborts the request when the consumer breaks out", async () => {
    let signal: AbortSignal | undefined;
    const fetch = fakeFetch((_url, init) => {
      signal = init.signal ?? undefined;
      return sseResponse([frame(0, started), frame(1, phase), frame(2, finished)]);
    });
    const client = new AgenticSearchClient({ baseUrl: "http://svc", fetch });
    for await (const _ of client.stream({ question: "q" })) break;
    expect(signal?.aborted).toBe(true);
  });

  it("propagates a caller abort to the request", async () => {
    let signal: AbortSignal | undefined;
    const fetch = fakeFetch((_url, init) => {
      signal = init.signal ?? undefined;
      return sseResponse([frame(0, started), frame(1, phase), frame(2, finished)]);
    });
    const controller = new AbortController();
    const client = new AgenticSearchClient({ baseUrl: "http://svc", fetch });
    const it = client.stream({ question: "q" }, { signal: controller.signal });
    await it.next();
    controller.abort();
    expect(signal?.aborted).toBe(true);
    await it.return(undefined);
  });

  it("gets profiles, health and one-shot search", async () => {
    const fetch = fakeFetch((url) => {
      if (url.endsWith("/v1/profiles")) return Response.json({ profiles: [{ name: "demo" }] });
      if (url.endsWith("/healthz")) return Response.json({ status: "ok", version: "0.1.0" });
      return Response.json(finished.result);
    });
    const client = new AgenticSearchClient({ baseUrl: "http://svc", fetch });
    expect((await client.profiles())[0]?.name).toBe("demo");
    expect((await client.health()).status).toBe("ok");
    expect((await client.search({ question: "q" })).stop_reason).toBe("no_plan");
  });
});
