import { describe, expect, it } from "vitest";
import { parseSse, type SseMessage } from "../src/sse.js";
import { chunked, streamOf } from "./helpers.js";

async function collect(chunks: string[]): Promise<SseMessage[]> {
  const out: SseMessage[] = [];
  for await (const m of parseSse(streamOf(chunks))) out.push(m);
  return out;
}

const BODY =
  ': keep-alive\n\n' +
  'id: 0\nevent: search_started\ndata: {"type":"search_started"}\n\n' +
  'id: 1\nevent: phase_started\ndata: {"a":\ndata: 1}\n\n';

describe("parseSse", () => {
  it("parses events, ids, comments and multi-line data", async () => {
    expect(await collect([BODY])).toEqual([
      { event: "search_started", data: '{"type":"search_started"}', id: "0" },
      { event: "phase_started", data: '{"a":\n1}', id: "1" },
    ]);
  });

  it("is independent of chunk boundaries", async () => {
    const whole = await collect([BODY]);
    for (const size of [1, 2, 3, 7, 13]) expect(await collect(chunked(BODY, size))).toEqual(whole);
  });

  it("accepts CRLF and CR line endings, split anywhere", async () => {
    const crlf = BODY.replaceAll("\n", "\r\n");
    const cr = BODY.replaceAll("\n", "\r");
    const whole = await collect([BODY]);
    for (const size of [1, 2, 5]) {
      expect(await collect(chunked(crlf, size))).toEqual(whole);
      expect(await collect(chunked(cr, size))).toEqual(whole);
    }
  });

  it("strips a BOM, defaults the event name and drops an unterminated message", async () => {
    expect(await collect(["﻿data: x\n\n", "data: never dispatched"])).toEqual([
      { event: "message", data: "x", id: null },
    ]);
  });

  it("handles fields without a space after the colon and without a value", async () => {
    expect(await collect(["event:e\ndata:v\ndata\n\n"])).toEqual([
      { event: "e", data: "v\n", id: null },
    ]);
  });

  it("decodes multi-byte UTF-8 split across chunks", async () => {
    const text = "data: café ✓\n\n";
    const bytes = new TextEncoder().encode(text);
    const stream = new ReadableStream<Uint8Array>({
      start(c) {
        for (let i = 0; i < bytes.length; i++) c.enqueue(bytes.slice(i, i + 1));
        c.close();
      },
    });
    const out: SseMessage[] = [];
    for await (const m of parseSse(stream)) out.push(m);
    expect(out[0]?.data).toBe("café ✓");
  });

  it("cancels the body when the consumer stops early", async () => {
    let cancelled = false;
    const stream = new ReadableStream<Uint8Array>({
      start(c) {
        c.enqueue(new TextEncoder().encode("data: 1\n\ndata: 2\n\n"));
      },
      cancel() {
        cancelled = true;
      },
    });
    for await (const _ of parseSse(stream)) break;
    expect(cancelled).toBe(true);
  });
});
