// Tests for the viewer's pure logic: the block between /* <core> */ and /* </core> */ in
// viewer/index.html, run in a VM context without a DOM.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

const html = readFileSync(new URL("../viewer/index.html", import.meta.url), "utf8");
const core = html.split("/* <core> */")[1].split("/* </core> */")[0];
const ctx = {};
vm.createContext(ctx);
vm.runInContext(core + "\n;globalThis.core = { SOURCES, parseSse, initialState, reduce, notesFromTrace, parseJsonl, schedule, assignFiles, overtime, clockSeconds, questionText, replayable };", ctx);
const { SOURCES, parseSse, initialState, reduce, notesFromTrace, parseJsonl, schedule, assignFiles, overtime, clockSeconds, questionText, replayable } = ctx.core;

const ev = (type, extra = {}) => ({ type, search_id: "s", seq: 0, turn: 0, at_ms: 0, ...extra });
const frame = (e) => `id: ${e.seq}\nevent: ${e.type}\ndata: ${JSON.stringify(e)}\n\n`;

test("SOURCES are in swimlane order", () => {
  assert.deepEqual([...SOURCES], ["incidents", "deploys", "pull_requests", "dashboards", "runbooks"]);
});

test("parseSse splits frames and keeps the partial tail", () => {
  const a = ev("phase_started", { phase: "plan" });
  const b = ev("phase_started", { phase: "query", seq: 1 });
  const half = frame(ev("phase_started", { phase: "judge", seq: 2 })).slice(0, 20);
  const { events, rest } = parseSse(frame(a) + frame(b) + half);
  assert.deepEqual([...events.map((e) => e.phase)], ["plan", "query"]);
  assert.equal(rest, half);
});

test("parseSse ignores keep-alive comments and CRLF endings", () => {
  const a = ev("phase_started", { phase: "plan" });
  const { events, rest } = parseSse(": keep-alive\n\n" + frame(a).replaceAll("\n", "\r\n"));
  assert.equal(events.length, 1);
  assert.equal(events[0].phase, "plan");
  assert.equal(rest, "");
});

test("reduce tracks the current phase", () => {
  const s = reduce(initialState("harness"), ev("phase_started", { phase: "judge", turn: 1 }));
  assert.equal(s.phase, "judge");
  assert.equal(s.turn, 1);
});

test("reduce records a tool call in its source row", () => {
  let s = reduce(initialState("harness"), ev("tool_call_started", {
    call_id: "c1", source: "deploys", name: "filter_search", arguments: { source: "deploys" }, turn: 1, at_ms: 1500 }));
  s = reduce(s, ev("tool_call_finished", { call_id: "c1", n_hits: 3, duration_ms: 5, error: null }));
  assert.equal(s.calls.length, 1);
  assert.equal(s.calls[0].source, "deploys");
  assert.equal(s.calls[0].name, "filter_search");
  assert.equal(s.calls[0].turn, 1);
  assert.equal(s.calls[0].nHits, 3);
});

test("reduce records a failed tool call", () => {
  let s = reduce(initialState("harness"), ev("tool_call_started", { call_id: "c1", source: "deploys", name: "lexical_search" }));
  s = reduce(s, ev("tool_call_finished", { call_id: "c1", n_hits: 0, duration_ms: 1, error: { kind: "validation", message: "bad" } }));
  assert.equal(s.calls[0].error, "validation: bad");
});

test("reduce records decide notes", () => {
  const s = reduce(initialState("harness"), ev("phase_finished", {
    phase: "decide", turn: 1, duration_ms: 1, summary: { action: "refine", note: "fix the errors" } }));
  assert.equal(s.decisions.length, 1);
  assert.equal(s.decisions[0].action, "refine");
  assert.equal(s.decisions[0].note, "fix the errors");
  assert.equal(s.decisions[0].turn, 1);
});

test("reduce keeps the latest results and usage", () => {
  let s = reduce(initialState("retrieval"), ev("results_updated", { hits: [{ key: "incidents:INC-1", doc_id: "INC-1" }], pool_size: 1, n_relevant: 1 }));
  s = reduce(s, ev("usage_updated", { usage: { turns: 2, tool_calls: 7, cost_usd: 0.01 }, at_ms: 3000 }));
  assert.equal(s.hits[0].doc_id, "INC-1");
  assert.equal(s.metrics.turns, 2);
  assert.equal(s.metrics.toolCalls, 7);
  assert.equal(s.metrics.costUsd, 0.01);
  assert.equal(s.metrics.seconds, 3);
});

test("reduce finishes with the stop reason and final usage", () => {
  const s = reduce(initialState("harness"), ev("search_finished", {
    stop_reason: "controller_stop", at_ms: 9000,
    result: { usage: { turns: 3, tool_calls: 18, cost_usd: 0.015 }, hits: [] } }));
  assert.equal(s.done, true);
  assert.equal(s.metrics.stopReason, "controller_stop");
  assert.equal(s.metrics.turns, 3);
});

test("reduce handles search_failed", () => {
  const s = reduce(initialState("harness"), ev("search_failed", { error_type: "APITimeoutError", message: "timed out" }));
  assert.equal(s.done, true);
  assert.equal(s.error, "APITimeoutError: timed out");
  assert.equal(s.metrics.stopReason, "failed");
});

test("reduce ignores unknown event types", () => {
  const before = initialState("harness");
  assert.deepEqual(reduce(before, ev("future_thing", { x: 1 })), before);
});

const traced = (notes) => ({ usage: { turns: 2, tool_calls: 3, cost_usd: 0 }, hits: [], trace: { events:
  notes.map((note, turn) => ({ type: "decision", turn, data: { action: "continue", note } })) } });

test("notesFromTrace maps turns to decision notes", () => {
  assert.deepEqual({ ...notesFromTrace(traced(["a", "b"])) }, { 0: "a", 1: "b" });
  assert.deepEqual({ ...notesFromTrace({ usage: {} }) }, {});
});

test("search_finished fills decision notes the events left empty", () => {
  let s = reduce(initialState("harness"), ev("phase_finished", {
    phase: "decide", turn: 0, duration_ms: 1, summary: { action: "continue", note: null } }));
  s = reduce(s, ev("search_finished", { stop_reason: "controller_stop", result: traced(["keep exploring"]) }));
  assert.equal(s.decisions[0].note, "keep exploring");
});

test("known notes are shown as soon as the decision arrives", () => {
  const start = { ...initialState("harness"), notes: { 0: "keep exploring" } };
  const s = reduce(start, ev("phase_finished", {
    phase: "decide", turn: 0, duration_ms: 1, summary: { action: "continue", note: null } }));
  assert.equal(s.decisions[0].note, "keep exploring");
});

test("parseJsonl skips bad and unknown lines", () => {
  const a = ev("phase_started", { phase: "plan" });
  const b = ev("search_finished", { seq: 1, stop_reason: "single_pass", result: {} });
  const text = [JSON.stringify(a), "{bad", "", JSON.stringify({ type: "future_thing" }), JSON.stringify(b)].join("\n");
  assert.deepEqual([...parseJsonl(text).map((e) => e.type)], ["phase_started", "search_finished"]);
});

test("schedule keeps relative timing and scales by speed", () => {
  const evs = [0, 100, 300].map((at_ms) => ev("phase_started", { at_ms }));
  assert.deepEqual([...schedule(evs, 2).map((x) => x.delayMs)], [0, 50, 150]);
  assert.deepEqual([...schedule(evs, 1).map((x) => x.delayMs)], [0, 100, 300]);
});

test("assignFiles matches files by suffix", () => {
  const files = [
    { name: "20261029-1200-hero-harness.jsonl", text: "h" },
    { name: "20261029-1200-hero.meta.json", text: '{"gold": ["INC-4821"], "text": "Q?"}' },
    { name: "20261029-1200-hero-retrieval.jsonl", text: "r" },
  ];
  const got = assignFiles(files);
  assert.equal(got.retrieval, "r");
  assert.equal(got.harness, "h");
  assert.deepEqual([...got.meta.gold], ["INC-4821"]);
});

test("assignFiles works without meta", () => {
  const got = assignFiles([{ name: "x-retrieval.jsonl", text: "r" }, { name: "x-harness.jsonl", text: "h" }]);
  assert.equal(got.meta, null);
  assert.equal(got.harness, "h");
});

test("overtime flags a live column still running past 60 s", () => {
  assert.equal(overtime(61, false, false), true);
  assert.equal(overtime(59, false, false), false);
  assert.equal(overtime(90, true, false), false);   // finished
  assert.equal(overtime(90, false, true), false);   // replay
});

test("clockSeconds: a paused replay keeps the replay clock", () => {
  const s = { ...initialState("harness"), metrics: { ...initialState("harness").metrics, seconds: 4 } };
  assert.equal(clockSeconds(s, { replay: true, vt: 6500 }), 6.5);
});

test("clockSeconds: live uses wall time since start; finished uses the final time", () => {
  const s = initialState("harness");
  assert.equal(clockSeconds(s, { replay: false, startedMs: 1000, nowMs: 3500 }), 2.5);
  const done = { ...s, done: true, metrics: { ...s.metrics, seconds: 9 } };
  assert.equal(clockSeconds(done, { replay: true, vt: 20000 }), 9);
  assert.equal(clockSeconds(s, { replay: false, startedMs: null, nowMs: 5 }), 0);
});

test("questionText prefers meta, then search_started, and never throws", () => {
  const started = { type: "search_started", question: { content: [{ kind: "text", text: "Q from run?" }] } };
  assert.equal(questionText({ text: "Q from meta?" }, [started]), "Q from meta?");
  assert.equal(questionText(null, [started]), "Q from run?");
  assert.equal(questionText(null, [{ type: "search_started", question: "plain string?" }]), "plain string?");
  assert.equal(questionText(null, [{ type: "search_started" }]), "");
  assert.equal(questionText(null, []), "");
});

test("replayable needs at least one run file", () => {
  assert.equal(replayable({ retrieval: "r", harness: null, meta: null }), true);
  assert.equal(replayable({ retrieval: null, harness: null, meta: { gold: [] } }), false);
});
