/** End to end against the real Python service (needs `uv` and the repo's Python env). */
import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { AgenticSearchClient, AgenticSearchError, type SearchEvent } from "../src/index.js";

let server: ChildProcessWithoutNullStreams;
let output = "";
let client: AgenticSearchClient;

function waitFor(text: string, timeoutMs: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`timed out waiting for ${text}; output:\n${output}`)), timeoutMs);
    const check = () => {
      if (output.includes(text)) {
        clearTimeout(timer);
        server.stdout.off("data", check);
        resolve();
      }
    };
    server.stdout.on("data", check);
    check();
  });
}

beforeAll(async () => {
  const cwd = fileURLToPath(new URL(".", import.meta.url));
  server = spawn("uv", ["run", "python", "serve_demo.py"], { cwd });
  server.stdout.on("data", (chunk: Buffer) => { output += chunk.toString(); });
  server.stderr.on("data", (chunk: Buffer) => { output += chunk.toString(); });
  await waitFor("LISTENING", 60_000);
  const port = /LISTENING (\d+)/.exec(output)![1];
  client = new AgenticSearchClient({ baseUrl: `http://127.0.0.1:${port}` });
}, 70_000);

afterAll(() => {
  server?.kill();
});

describe("against the real service", () => {
  it("reports health and profiles", async () => {
    expect((await client.health()).status).toBe("ok");
    const profiles = await client.profiles();
    expect(profiles.map((p) => p.name).sort()).toEqual(["demo", "slow"]);
    expect(profiles[0]!.sources[0]!.capabilities).toContain("lexical");
  });

  it("streams a search to search_finished", async () => {
    const events: SearchEvent[] = [];
    for await (const ev of client.stream({ profile: "demo", question: "what treats headache", snapshot_k: 3 })) {
      events.push(ev);
    }
    expect(events[0]!.type).toBe("search_started");
    expect(events.map((e) => e.seq)).toEqual(events.map((_, i) => i));
    const last = events.at(-1)!;
    if (last.type !== "search_finished") throw new Error(`ended with ${last.type}`);
    expect(last.result.hits.map((h) => h.hit.doc_id).sort()).toEqual(["d1", "d4"]);
    expect(last.result.trace).toBeUndefined();
  });

  it("returns the same hits from one-shot search", async () => {
    const result = await client.search({ profile: "demo", question: "what treats headache" });
    expect(result.hits.map((h) => h.hit.doc_id).sort()).toEqual(["d1", "d4"]);
  });

  it("delivers a failed search as an event, and HTTP errors as AgenticSearchError", async () => {
    const events: SearchEvent[] = [];
    for await (const ev of client.stream({ profile: "demo", question: "q", sources: ["nope"] })) events.push(ev);
    expect(events.map((e) => e.type)).toEqual(["search_failed"]);
    await expect(client.search({ profile: "missing", question: "q" })).rejects.toMatchObject({ status: 404 });
    await expect(client.search({ profile: "demo", question: "" })).rejects.toBeInstanceOf(AgenticSearchError);
  });

  it("cancels the search on the server when the consumer stops", async () => {
    const before = (output.match(/CANCELLED/g) ?? []).length;
    for await (const ev of client.stream({ profile: "slow", question: "headache" })) {
      if (ev.type === "tool_call_started") break;
    }
    const deadline = Date.now() + 10_000;
    while ((output.match(/CANCELLED/g) ?? []).length === before) {
      if (Date.now() > deadline) throw new Error(`server never cancelled; output:\n${output}`);
      await new Promise((r) => setTimeout(r, 50));
    }
  }, 15_000);

  it("cancels the search when the caller aborts", async () => {
    const before = (output.match(/CANCELLED/g) ?? []).length;
    const controller = new AbortController();
    const seen: string[] = [];
    await expect((async () => {
      for await (const ev of client.stream({ profile: "slow", question: "headache" }, { signal: controller.signal })) {
        seen.push(ev.type);
        if (ev.type === "tool_call_started") controller.abort();
      }
    })()).rejects.toMatchObject({ name: "AbortError" });
    expect(seen.at(-1)).toBe("tool_call_started");
    const deadline = Date.now() + 10_000;
    while ((output.match(/CANCELLED/g) ?? []).length === before) {
      if (Date.now() > deadline) throw new Error(`server never cancelled; output:\n${output}`);
      await new Promise((r) => setTimeout(r, 50));
    }
  }, 15_000);
});
