/** End to end against the real Python service (needs `uv` and the repo's Python env). */
import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  AgenticSearchClient,
  AgenticSearchError,
  type Health,
  type ProfileInfo,
  type SearchEvent,
  type SourceInfo,
} from "../src/index.js";

/** Compile-time: `L` must list every key of `T` and nothing else (as in contract.test.ts). */
type Exactly<T, L extends readonly (keyof T)[]> = Exclude<keyof T, L[number]> extends never ? L : never;
function fields<T>() {
  return <const L extends readonly (keyof T)[]>(list: Exactly<T, L>) => list;
}
const PROFILE_FIELDS = fields<ProfileInfo>()(["name", "default", "available", "error", "mode", "budget", "limits", "sources", "setup_errors"]);
const SOURCE_FIELDS = fields<SourceInfo>()(["name", "backend_type", "capabilities", "collections", "description"]);
const HEALTH_FIELDS = fields<Health>()(["status", "version"]);
const sorted = (xs: Iterable<string>) => [...xs].sort();

let server: ChildProcessWithoutNullStreams;
let output = "";
let client: AgenticSearchClient;

function waitFor(text: string, timeoutMs: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const finish = (error?: Error) => {
      clearTimeout(timer);
      server.stdout.off("data", check);
      server.off("error", onError);
      server.off("exit", onExit);
      if (error) reject(error); else resolve();
    };
    const timer = setTimeout(() => finish(new Error(`timed out waiting for ${text}; output:\n${output}`)), timeoutMs);
    const check = () => {
      if (output.includes(text)) finish();
    };
    const onError = (err: Error) => finish(new Error(`server failed to start: ${err.message}; output:\n${output}`));
    const onExit = (code: number | null) => {
      if (!output.includes(text)) finish(new Error(`server exited (${code}) before ${text}; output:\n${output}`));
    };
    server.stdout.on("data", check);
    server.on("error", onError);
    server.on("exit", onExit);
    check();
  });
}

async function waitForCancel(before: number): Promise<void> {
  const deadline = Date.now() + 10_000;
  while ((output.match(/CANCELLED/g) ?? []).length === before) {
    if (Date.now() > deadline) throw new Error(`server never cancelled; output:\n${output}`);
    await new Promise((r) => setTimeout(r, 50));
  }
}

beforeAll(async () => {
  const cwd = fileURLToPath(new URL(".", import.meta.url));
  server = spawn("uv", ["run", "python", "serve_demo.py"], { cwd, detached: true });
  server.stdout.on("data", (chunk: Buffer) => { output += chunk.toString(); });
  server.stderr.on("data", (chunk: Buffer) => { output += chunk.toString(); });
  await waitFor("LISTENING", 60_000);
  const port = /LISTENING (\d+)/.exec(output)![1];
  client = new AgenticSearchClient({ baseUrl: `http://127.0.0.1:${port}` });
}, 70_000);

afterAll(() => {
  if (server?.pid) {
    try { process.kill(-server.pid, "SIGTERM"); } catch { /* already gone */ }
  }
});

describe("against the real service", () => {
  it("reports health and profiles", async () => {
    const health = await client.health();
    expect(health.status).toBe("ok");
    expect(sorted(Object.keys(health))).toEqual(sorted(HEALTH_FIELDS));
    const profiles = await client.profiles();
    // ProfileInfo, SourceInfo and Health are hand-written (not in the event schema): drift fails here.
    for (const p of profiles) {
      expect(sorted(Object.keys(p)), p.name).toEqual(sorted(PROFILE_FIELDS));
      for (const s of p.sources) expect(sorted(Object.keys(s)), `${p.name}/${s.name}`).toEqual(sorted(SOURCE_FIELDS));
    }
    expect(profiles.some((p) => p.sources.length > 0)).toBe(true);
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
    await waitForCancel(before);
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
    await waitForCancel(before);
  }, 15_000);
});
