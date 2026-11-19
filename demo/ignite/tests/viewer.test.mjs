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
vm.runInContext(core + "\n;globalThis.core = { SETS, optionsHtml, requestBody, TOOLS, laneNames, chipLabel, parseSse, initialState, reduce, notesFromTrace, parseJsonl, schedule, assignFiles, overtime, clockSeconds, questionText, replayable, stepsHtml, goldSummary, snippetText, answerHtml, goldLabel, goldShare, tilesHtml, verdictHtml, markText, proofHtml, coverageHtml, MODES, initialSwarm, reduceSwarm, swarmRowsHtml, swarmRequestBody, swarmTimeline };", ctx);
const { SETS, optionsHtml, requestBody, TOOLS, laneNames, chipLabel, parseSse, initialState, reduce, notesFromTrace, parseJsonl, schedule, assignFiles, overtime, clockSeconds, questionText, replayable, stepsHtml, goldSummary, snippetText, answerHtml, goldLabel, goldShare, tilesHtml, verdictHtml, markText, proofHtml, coverageHtml, MODES, initialSwarm, reduceSwarm, swarmRowsHtml, swarmRequestBody, swarmTimeline } = ctx.core;

const ev = (type, extra = {}) => ({ type, search_id: "s", seq: 0, turn: 0, at_ms: 0, ...extra });
const frame = (e) => `id: ${e.seq}\nevent: ${e.type}\ndata: ${JSON.stringify(e)}\n\n`;

test("optionsHtml groups questions by set, in SETS order", () => {
  const html = optionsHtml([
    { id: "m1", text: "Match <it>", set: "wands-match" },
    { id: "hero", text: "Who?", set: "musique" },
    { id: "w1", text: "gray sofa", set: "wands" },
  ]);
  assert.deepEqual([...SETS.map((s) => s.id)], ["musique", "wands", "wands-match"]);
  const groups = [...html.matchAll(/<optgroup label="([^"]+)">/g)].map((m) => m[1]);
  assert.deepEqual(groups, ["MuSiQue (multi-hop Wikipedia)", "WANDS (product search)", "WANDS match (our questions)"]);
  assert.ok(html.indexOf('value="hero"') < html.indexOf('value="w1"') && html.indexOf('value="w1"') < html.indexOf('value="m1"'));
  assert.ok(html.includes("Match &lt;it&gt;"));
});

test("optionsHtml leaves out empty sets", () => {
  assert.ok(!optionsHtml([{ id: "hero", text: "Q", set: "musique" }]).includes("WANDS"));
});

test("a live run posts the question's profile", () => {
  const body = JSON.parse(requestBody({ id: "w1", text: "gray sofa", profile: "ignite-wands" }, "harness"));
  assert.deepEqual(body, { profile: "ignite-wands", question: "gray sofa", mode: "harness", snapshot_k: 10, include_trace: true });
});

test("lanes are the known tools in order, then any other tool seen", () => {
  assert.deepEqual([...TOOLS], ["lexical_search", "vector_search", "filter_search", "fetch"]);
  assert.deepEqual([...laneNames([{ name: "fetch" }, { name: "zoom" }, { name: "zoom" }])],
    ["lexical_search", "vector_search", "filter_search", "fetch", "zoom"]);
});

test("chipLabel shows what the call searched for", () => {
  assert.equal(chipLabel({ text: "Battle of Hastings", source: "wiki" }), "Battle of Hastings");
  assert.equal(chipLabel({ hyde_text: "a paragraph about X", field: "embedding" }), "a paragraph about X");
  assert.equal(chipLabel({ doc_ids: ["msq-a", "msq-b"], collection: "wiki" }), "msq-a, msq-b");
  assert.equal(chipLabel({ filter: { field: "title", op: "eq", value: "Paris" }, source: "wiki" }), "title = Paris");
  assert.equal(chipLabel({ pattern: "Columbia.*border", collection: "wiki" }), "/Columbia.*border/");
  assert.equal(chipLabel({}), "");
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
  let s = reduce(initialState("retrieval"), ev("results_updated", { hits: [{ key: "wiki:msq-1", doc_id: "msq-1" }], pool_size: 1, n_relevant: 1 }));
  s = reduce(s, ev("usage_updated", { usage: { turns: 2, tool_calls: 7, cost_usd: 0.01 }, at_ms: 3000 }));
  assert.equal(s.hits[0].doc_id, "msq-1");
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
    { name: "20261029-1200-hero.meta.json", text: '{"gold": ["msq-gold"], "text": "Q?"}' },
    { name: "20261029-1200-hero-retrieval.jsonl", text: "r" },
  ];
  const got = assignFiles(files);
  assert.equal(got.retrieval, "r");
  assert.equal(got.harness, "h");
  assert.deepEqual([...got.meta.gold], ["msq-gold"]);
});

test("assignFiles works without meta", () => {
  const got = assignFiles([{ name: "x-retrieval.jsonl", text: "r" }, { name: "x-harness.jsonl", text: "h" }]);
  assert.equal(got.meta, null);
  assert.equal(got.harness, "h");
});

test("overtime flags a live column still running past 120 s", () => {
  assert.equal(overtime(121, false, false), true);
  assert.equal(overtime(119, false, false), false);
  assert.equal(overtime(150, true, false), false);  // finished
  assert.equal(overtime(150, false, true), false);  // replay
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

test("reduce keeps steps and facts", () => {
  let s = initialState("harness");
  assert.equal(JSON.stringify([s.steps, s.current, s.facts]), JSON.stringify([[], null, []]));
  const steps = [{ text: "find author", done: true }, { text: "find city", done: false }];
  s = reduce(s, ev("steps_updated", { steps, current: 1 }));
  assert.equal(JSON.stringify(s.steps), JSON.stringify(steps));
  assert.equal(s.current, 1);
  s = reduce(s, ev("fact_noted", { fact: "A wrote B", source_key: "wiki:1" }));
  s = reduce(s, ev("fact_noted", { fact: "B is in C", source_key: "wiki:2" }));
  assert.equal(JSON.stringify(s.facts), JSON.stringify([{ fact: "A wrote B", source_key: "wiki:1" }, { fact: "B is in C", source_key: "wiki:2" }]));
  s = reduce(s, ev("steps_updated", { steps: steps.map((x) => ({ ...x, done: true })), current: null }));
  assert.equal(s.current, null);
  assert.equal(s.facts.length, 2);
});

test("replay keeps steps and facts events", () => {
  const text = [ev("steps_updated", { steps: [], current: null }), ev("fact_noted", { fact: "f", source_key: "k" })]
    .map((e) => JSON.stringify(e)).join("\n");
  assert.equal(parseJsonl(text).length, 2);
});

test("stepsHtml marks the current step", () => {
  assert.equal(stepsHtml(initialState("harness")), "");
  let s = reduce(initialState("harness"), ev("steps_updated", {
    steps: [{ text: "find <author>", done: true }, { text: "find city", done: false }, { text: "answer", done: false }],
    current: 1 }));
  s = reduce(s, ev("fact_noted", { fact: "A wrote B", source_key: "wiki:1" }));
  const html = stepsHtml(s);
  assert.equal((html.match(/class="step[ "]/g) || []).length, 3);
  assert.equal((html.match(/✓/g) || []).length, 1);
  assert.equal((html.match(/step current/g) || []).length, 1);
  assert.equal((html.match(/>now</g) || []).length, 1);
  assert.ok(html.indexOf("step current") > html.indexOf("✓"));
  assert.ok(html.includes("find &lt;author&gt;"));
  assert.ok(html.includes("A wrote B") && html.includes("wiki:1"));
  const allDone = reduce(s, ev("steps_updated", { steps: s.steps.map((x) => ({ ...x, done: true })), current: null }));
  assert.ok(!stepsHtml(allDone).includes(">now<"));
  assert.equal(stepsHtml({ ...initialState("harness"), facts: [{ fact: "only", source_key: "k" }] }).includes("only"), true);
});

test("goldSummary counts gold documents in the top 10 only", () => {
  const hits = Array.from({ length: 12 }, (_, i) => ({ doc_id: `d${i}` }));
  assert.deepEqual({ ...goldSummary(hits, ["d0", "d3", "d11", "zz"]) }, { inTop: 2, total: 4 });
  assert.deepEqual({ ...goldSummary([], []) }, { inTop: 0, total: 0 });
});

test("snippetText drops a repeated title and cuts long text at a word", () => {
  assert.equal(snippetText({ title: "Andrew Deveaux", snippet: "Andrew Deveaux Andrew Deveaux (1758) was a Loyalist." }),
               "Andrew Deveaux (1758) was a Loyalist.");
  const long = snippetText({ title: "", snippet: "word ".repeat(100) }, 50);
  assert.ok(long.endsWith(" …") && long.length <= 53);
  assert.equal(snippetText({}), "");
});

test("answerHtml shows pending, right, wrong and unchecked answers", () => {
  assert.equal(answerHtml(null), "");
  assert.ok(answerHtml({ pending: true }).includes("…"));
  const right = answerHtml({ answer: "Richland County.", correct: true, expected: "Richland County" });
  assert.ok(right.includes('class="answer right"') && right.includes("✓ correct"));
  const wrong = answerHtml({ answer: "Lexington <b>", correct: false, expected: "Richland County" });
  assert.ok(wrong.includes("✗ wrong") && wrong.includes("Richland County") && wrong.includes("Lexington &lt;b&gt;"));
  const plain = answerHtml({ answer: "Three teak sofas.", correct: null, expected: null });
  assert.ok(!plain.includes("✓") && !plain.includes("✗"));
  assert.ok(answerHtml({ error: "HTTP 500" }).includes("no answer"));
});

test("initialState has no answer yet", () => {
  assert.equal(initialState("harness").answer, null);
});

test("the whole page script compiles (not only the core block)", () => {
  const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]);
  assert.ok(scripts.length >= 1);
  for (const src of scripts) new vm.Script(src); // throws SyntaxError, e.g. on a duplicate const
});

test("answerHtml lists the reader's steps without their own numbering", () => {
  const html = answerHtml({ answer: "Richland County", correct: true, expected: "Richland County",
                            steps: ["Step 1: from South Carolina [2]", "2. Columbia is the capital [3]"] });
  assert.ok(html.includes('<ol class="why">'));
  assert.ok(html.includes("<li>from South Carolina [2]</li>") && html.includes("<li>Columbia is the capital [3]</li>"));
});

test("goldLabel counts needed docs for small gold sets and right results for big ones", () => {
  assert.equal(goldLabel({ inTop: 4, total: 4 }), "4 of 4 needed docs found");
  assert.equal(goldLabel({ inTop: 8, total: 34 }), "8 of 10 results right");
  assert.equal(goldShare({ inTop: 2, total: 4 }), 0.5);
  assert.equal(goldShare({ inTop: 7, total: 34 }), 0.7);
  assert.equal(goldShare({ inTop: 0, total: 0 }), 0);
});

test("tilesHtml: empty, unscored, hit and miss squares", () => {
  const hits = [{ doc_id: "a" }, { doc_id: "b" }];
  const unscored = tilesHtml(hits, null);
  assert.equal((unscored.match(/tile found/g) || []).length, 2);
  assert.equal((unscored.match(/class="tile"/g) || []).length, 8);
  const many = new Set(["b", ...Array.from({ length: 12 }, (_, i) => `g${i}`)]);
  const scored = tilesHtml(hits, many);
  assert.equal((scored.match(/tile hit/g) || []).length, 1);
  assert.equal((scored.match(/tile miss/g) || []).length, 1);
  const few = tilesHtml(hits, new Set(["b", "c"]));
  assert.equal((few.match(/tile other/g) || []).length, 1);
  assert.ok(!few.includes("tile miss"));
});

test("verdictHtml names the winner and the cost, and waits for both modes", () => {
  const r = { g: { inTop: 0, total: 34 }, seconds: 16, costUsd: 0.02 };
  const h = { g: { inTop: 10, total: 34 }, seconds: 44, costUsd: 0.06 };
  const html = verdictHtml(r, h);
  assert.ok(html.includes("The agent loop found more."));
  assert.ok(html.includes("Harness: 10 of 10 results right.") && html.includes("One pass: 0 of 10 results right."));
  assert.ok(html.includes("2.8× the time") && html.includes("3.0× the cost"));
  assert.equal(verdictHtml(null, h), "");
  assert.ok(verdictHtml(h, h).includes("A tie"));
});

test("markText escapes the text and marks each phrase, any case and spacing", () => {
  const html = markText("Style :  Farmhouse / Country <b>|color : gray", ["style : farmhouse / country", "(x)"]);
  assert.equal(html, "<mark>Style :  Farmhouse / Country</mark> &lt;b&gt;|color : gray");
  assert.equal(markText("a < b", []), "a &lt; b");
});

test("proofHtml says why, then shows the document text", () => {
  const html = proofHtml({ verdict: "right", why: "Catalogue: type Faucets ✓", text: "style : modern", marks: ["style : modern"] });
  assert.ok(html.includes('class="proof v-right"') && html.includes("★ Right.") && html.includes("<mark>style : modern</mark>"));
  assert.ok(proofHtml({ verdict: "odd", why: "x", text: "" }).includes("v-wrong"));
  assert.equal(proofHtml(null), "");
});

test("coverageHtml lists right results per type and flags a type with none", () => {
  const html = coverageHtml([["Bathroom Sink Faucets", 9], ["Towel Bars", 0]]);
  assert.ok(html.includes("Bathroom Sink Faucets: 9") && html.includes('<span class="zero">Towel Bars: 0</span>'));
  assert.equal(coverageHtml([["One", 3]]), "");
  assert.equal(coverageHtml(null), "");
});

// ---- swarm lane ----
const split = (subs, extra = {}) => ({ type: "swarm_split", sub_questions: subs, fallback: false, at_ms: 1000, ...extra });
const agentEv = (agent, event) => ({ type: "agent_event", agent, event });
const mhit = (key, agents = [0]) => ({ key, source: "wiki", doc_id: key.split(":")[1], title: key, snippet: "", score: 0.03, agents });

test("MODES lists the three lanes in order", () => {
  assert.deepEqual([...MODES], ["retrieval", "harness", "swarm"]);
});

test("reduceSwarm folds agent events into each agent", () => {
  let s = initialSwarm();
  assert.equal(s.mode, "swarm");
  s = reduceSwarm(s, { type: "swarm_split", sub_questions: ["a", "b"], fallback: false });
  s = reduceSwarm(s, { type: "agent_event", agent: 1, event: ev("search_started", { mode: "harness" }) });
  assert.equal(s.agents.length, 2); assert.equal(s.agents[1].phase !== undefined, true);
  assert.deepEqual([...s.subs], ["a", "b"]);
  s = reduceSwarm(s, agentEv(1, ev("phase_started", { phase: "query", turn: 1 })));
  assert.equal(s.agents[1].phase, "query");
  assert.equal(s.agents[1].turn, 1);
  assert.equal(s.agents[0].phase, null);  // the other agent is untouched
  assert.equal(s.agents[0].status, "running");
});

test("reduceSwarm: an agent's own finish or failure sets its status", () => {
  let s = reduceSwarm(initialSwarm(), split(["a", "b"]));
  s = reduceSwarm(s, agentEv(0, ev("search_finished", { stop_reason: "done", result: { usage: { turns: 2, tool_calls: 3, cost_usd: 0.01 } } })));
  s = reduceSwarm(s, agentEv(1, { type: "search_failed", error_type: "Boom", message: "x" }));
  assert.equal(s.agents[0].status, "done");
  assert.equal(s.agents[1].status, "failed");
  assert.equal(s.done, false);  // the swarm is done only at swarm_finished
});

test("reduceSwarm ignores an agent index it has no row for", () => {
  const s = reduceSwarm(initialSwarm(), split(["a"]));
  assert.equal(reduceSwarm(s, agentEv(5, ev("search_started"))), s);
});

test("swarm_finished sets merged hits, usage and agent status", () => {
  let s = reduceSwarm(initialSwarm(), split(["a", "b"]));
  s = reduceSwarm(s, { type: "swarm_finished", hits: [mhit("wiki:x", [0, 1]), mhit("wiki:y", [1])],
    notes: ["X was born in Y"], usage: { input_tokens: 1, output_tokens: 2, cost_usd: 0.2, tool_calls: 7, turns: 2 },
    seconds: 34.2, agent_status: ["done", "cut"], at_ms: 34200 });
  assert.equal(s.done, true);
  assert.equal(s.hits.length, 2);
  assert.equal(s.agents[1].status, "cut");
  assert.equal(s.agents[0].status, "done");
  assert.equal(s.usage.cost_usd, 0.2);
  assert.equal(s.seconds, 34.2);
  assert.deepEqual([...s.notes], ["X was born in Y"]);
});

test("swarm_failed ends the lane with an error", () => {
  const s = reduceSwarm(initialSwarm(), { type: "swarm_failed", error_type: "RuntimeError", message: "no model", at_ms: 50 });
  assert.equal(s.done, true);
  assert.equal(s.error, "RuntimeError: no model");
});

test("swarm_split records the fallback flag", () => {
  const s = reduceSwarm(initialSwarm(), split(["whole question"], { fallback: true }));
  assert.equal(s.fallback, true);
  assert.equal(s.agents.length, 1);
});

test("swarmRowsHtml shows one row per agent: index, sub-question, turn dots, hits, status", () => {
  let s = reduceSwarm(initialSwarm(), split(["Where was A born?", "Capital of that state?"]));
  s = reduceSwarm(s, agentEv(0, ev("phase_started", { phase: "plan", turn: 1 })));
  s = reduceSwarm(s, agentEv(0, ev("results_updated", { hits: [mhit("wiki:a"), mhit("wiki:b"), mhit("wiki:c")] })));
  const html = swarmRowsHtml(s);
  assert.equal((html.match(/<tr/g) || []).length, 2);
  assert.ok(html.includes(">1<") && html.includes(">2<"));
  assert.ok(html.includes("Where was A born?"));
  assert.ok(html.includes("●○"));        // turn 1 done, turn 2 running
  assert.ok(html.includes(">3<"));       // hit count
  assert.ok(html.includes("running"));
});

test("swarmRowsHtml truncates a long sub-question and keeps it whole in the tooltip", () => {
  const long = "word ".repeat(40).trim();
  const html = swarmRowsHtml(reduceSwarm(initialSwarm(), split([long])));
  assert.ok(html.includes(`title="${long}"`));
  assert.ok(html.includes("…"));
});

test("swarmRowsHtml escapes sub-questions", () => {
  const html = swarmRowsHtml(reduceSwarm(initialSwarm(), split(["<b>bold</b>"])));
  assert.ok(html.includes("&lt;b&gt;") && !html.includes("<b>bold"));
});

test("swarmRowsHtml is empty before the split", () => {
  assert.equal(swarmRowsHtml(initialSwarm()), "");
});

test("swarmRequestBody posts profile and question, no mode", () => {
  const body = JSON.parse(swarmRequestBody({ id: "w1", text: "gray sofa", profile: "ignite-wands" }));
  assert.deepEqual(body, { profile: "ignite-wands", question: "gray sofa", snapshot_k: 10, include_trace: true });
});

test("assignFiles picks up a -swarm.jsonl", () => {
  const out = assignFiles([{ name: "x-swarm.jsonl", text: "S" }, { name: "x-harness.jsonl", text: "h" }]);
  assert.equal(out.swarm, "S");
  assert.equal(out.harness, "h");
  assert.equal(assignFiles([{ name: "x-harness.jsonl", text: "h" }]).swarm, null);
});

test("parseJsonl keeps swarm wire events", () => {
  const text = [JSON.stringify({ type: "swarm_started", question: "Q", at_ms: 0 }),
    JSON.stringify(agentEv(0, ev("search_started"))), JSON.stringify({ type: "swarm_finished", hits: [], at_ms: 9 })].join("\n");
  assert.deepEqual([...parseJsonl(text).map((e) => e.type)], ["swarm_started", "agent_event", "swarm_finished"]);
});

test("a swarm replay file renders the lane end to end", () => {
  const lines = [{ type: "swarm_started", question: "Q?", max_agents: 4, at_ms: 0 }, split(["a", "b"]),
    agentEv(0, ev("search_started")), agentEv(1, ev("search_started")),
    agentEv(0, ev("results_updated", { hits: [mhit("wiki:a")] })),
    { type: "swarm_finished", hits: [mhit("wiki:a")], notes: [], usage: { cost_usd: 0.1, tool_calls: 2, turns: 1 },
      seconds: 3, agent_status: ["done", "failed"], at_ms: 3000 }];
  const evs = parseJsonl(lines.map((l) => JSON.stringify(l)).join("\n"));
  const s = evs.reduce(reduceSwarm, initialSwarm());
  assert.equal(s.done, true);
  assert.equal(questionText(null, evs), "Q?");
  const html = swarmRowsHtml(s);
  assert.ok(html.includes("done") && html.includes("failed"));
});

test("swarmTimeline puts agent events on the swarm clock and keeps it monotonic", () => {
  const evs = [{ type: "swarm_started", at_ms: 0 }, split(["a", "b"], { at_ms: 1000 }),
    agentEv(0, ev("search_started", { at_ms: 1 })), agentEv(1, ev("phase_started", { at_ms: 500 })),
    agentEv(0, ev("phase_started", { at_ms: 200 })), { type: "swarm_finished", at_ms: 3000 }];
  const t = swarmTimeline(evs);
  assert.deepEqual([...t.map((e) => e.at_ms)], [0, 1000, 1001, 1500, 1500, 3000]);
  assert.equal(t[2].event.at_ms, 1);   // the agent's own event is unchanged
  assert.equal(evs[2].at_ms, undefined); // input not mutated
});

test("replayable accepts a swarm-only run", () => {
  assert.equal(replayable({ retrieval: null, harness: null, swarm: "s", meta: null }), true);
});

test("old two-lane replay: no swarm file, verdict still renders", () => {
  const r = { g: { inTop: 0, total: 34 }, seconds: 16, costUsd: 0.02 };
  const h = { g: { inTop: 10, total: 34 }, seconds: 44, costUsd: 0.06 };
  assert.equal(verdictHtml(r, h, null), verdictHtml(r, h));
  assert.ok(verdictHtml(r, h).length > 0);
  assert.ok(!verdictHtml(r, h).includes("Swarm"));
});

test("verdictHtml adds a swarm clause and names the best of three", () => {
  const r = { g: { inTop: 2, total: 34 }, seconds: 16, costUsd: 0.02 };
  const h = { g: { inTop: 5, total: 34 }, seconds: 44, costUsd: 0.06 };
  const s = { g: { inTop: 6, total: 34 }, seconds: 60, costUsd: 0.2 };
  const html = verdictHtml(r, h, s);
  assert.ok(html.includes("Swarm: 6 of 10 results right."));
  assert.ok(html.includes("The swarm found the most."));
  assert.ok(html.includes("The swarm took 3.8× the time and 10×"));
  assert.ok(verdictHtml(r, h, { ...s, g: h.g }).includes("The agent loop and the swarm tie for the most."));
  assert.ok(verdictHtml(s, s, s).includes("A tie on the results."));
});
