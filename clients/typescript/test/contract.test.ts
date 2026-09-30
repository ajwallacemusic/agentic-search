/**
 * The TypeScript types must match the Python event models. The schema and fixtures come from
 * `agentic-search export-schema` (committed under the repo's `schema/`). Field lists below are
 * checked two ways: at compile time they must name exactly the TS interface's keys, and at run
 * time they must equal the JSON Schema's properties. Together these catch drift on either side.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  EVENT_TYPES,
  SCHEMA_VERSION,
  isSearchEvent,
  type Action,
  type Budget,
  type Hit,
  type HitSummary,
  type ImagePart,
  type Mode,
  type OpRef,
  type Phase,
  type PhaseSummary,
  type Query,
  type RankedHit,
  type SearchEvent,
  type SearchResult,
  type StopReason,
  type StructuredPart,
  type TextPart,
  type ToolErrorInfo,
  type ToolErrorKind,
  type Trace,
  type TraceEvent,
  type Usage,
} from "../src/events.js";

const schemaDir = fileURLToPath(new URL("../../../schema/", import.meta.url));
const schema = JSON.parse(readFileSync(`${schemaDir}search-events.v${SCHEMA_VERSION}.schema.json`, "utf8"));
const fixtures: Record<string, { id: number; event: string; data: Record<string, unknown> }[]> =
  JSON.parse(readFileSync(`${schemaDir}search-events.v${SCHEMA_VERSION}.fixtures.json`, "utf8"));

/** Compile-time: `L` must list every key of `T` and nothing else. */
type Exactly<T, L extends readonly (keyof T)[]> = Exclude<keyof T, L[number]> extends never ? L : never;
function fields<T>() {
  return <const L extends readonly (keyof T)[]>(list: Exactly<T, L>) => list;
}

/** Compile-time: `L` must list every member of union `T` and nothing else. */
function allOf<T extends string>() {
  return <const L extends readonly T[]>(list: Exclude<T, L[number]> extends never ? L : never) => list;
}

type EventOf<K extends SearchEvent["type"]> = Extract<SearchEvent, { type: K }>;

const STOP_REASONS = allOf<StopReason>()(["controller_stop", "no_plan", "single_pass", "delegate_done", "budget_turns", "budget_tool_calls", "budget_tokens", "budget_cost", "budget_time"]);
const PHASES = allOf<Phase>()(["plan", "query", "judge", "decide", "delegate"]);
const MODES = allOf<Mode>()(["retrieval", "harness", "model"]);
const ACTIONS = allOf<Action>()(["continue", "refine", "broaden", "switch_source", "stop"]);
const TOOL_ERROR_KINDS = allOf<ToolErrorKind>()(["validation", "backend", "timeout", "embedder", "policy"]);

const EVENT_FIELDS = {
  search_started: fields<EventOf<"search_started">>()(["type", "schema_version", "search_id", "seq", "turn", "at_ms", "question", "mode", "sources", "budget", "setup_errors"]),
  phase_started: fields<EventOf<"phase_started">>()(["type", "schema_version", "search_id", "seq", "turn", "at_ms", "phase"]),
  phase_finished: fields<EventOf<"phase_finished">>()(["type", "schema_version", "search_id", "seq", "turn", "at_ms", "phase", "duration_ms", "summary"]),
  tool_call_started: fields<EventOf<"tool_call_started">>()(["type", "schema_version", "search_id", "seq", "turn", "at_ms", "call_id", "source", "name", "arguments"]),
  tool_call_finished: fields<EventOf<"tool_call_finished">>()(["type", "schema_version", "search_id", "seq", "turn", "at_ms", "call_id", "n_hits", "duration_ms", "error"]),
  results_updated: fields<EventOf<"results_updated">>()(["type", "schema_version", "search_id", "seq", "turn", "at_ms", "hits", "pool_size", "n_relevant"]),
  usage_updated: fields<EventOf<"usage_updated">>()(["type", "schema_version", "search_id", "seq", "turn", "at_ms", "usage"]),
  search_finished: fields<EventOf<"search_finished">>()(["type", "schema_version", "search_id", "seq", "turn", "at_ms", "result", "stop_reason"]),
  search_failed: fields<EventOf<"search_failed">>()(["type", "schema_version", "search_id", "seq", "turn", "at_ms", "error_type", "message"]),
} satisfies Record<SearchEvent["type"], readonly string[]>;

const MODEL_FIELDS: Record<string, readonly string[]> = {
  Budget: fields<Budget>()(["max_turns", "max_tool_calls", "max_tokens", "max_cost_usd", "max_seconds"]),
  Usage: fields<Usage>()(["input_tokens", "output_tokens", "cost_usd", "tool_calls", "turns"]),
  PhaseSummary: fields<PhaseSummary>()(["n_calls", "n_hits", "n_new", "n_errors", "n_judged", "n_relevant", "action", "confidence", "n_ranked", "note", "error"]),
  ToolErrorInfo: fields<ToolErrorInfo>()(["kind", "message", "source"]),
  HitSummary: fields<HitSummary>()(["key", "source", "doc_id", "title", "snippet", "score", "p_relevant", "judged", "first_turn", "content"]),
  Query: fields<Query>()(["content"]),
  TextPart: fields<TextPart>()(["kind", "text"]),
  ImagePart: fields<ImagePart>()(["kind", "uri", "data", "mime"]),
  StructuredPart: fields<StructuredPart>()(["kind", "data"]),
  OpRef: fields<OpRef>()(["turn", "call_id", "rank"]),
  Hit: fields<Hit>()(["doc_id", "source", "content", "metadata", "raw_score", "provenance"]),
  RankedHit: fields<RankedHit>()(["hit", "score", "p_relevant", "rationale", "judged"]),
  SearchResult: fields<SearchResult>()(["question", "hits", "stop_reason", "usage", "mode", "trace"]),
  Trace: fields<Trace>()(["events"]),
  TraceEvent: fields<TraceEvent>()(["type", "turn", "at_ms", "duration_ms", "data"]),
};

const sorted = (xs: Iterable<string>) => [...xs].sort();
const defOf = (ref: string) => schema.$defs[ref.replace("#/$defs/", "")];

describe("contract with the Python models (schema v1)", () => {
  it("knows exactly the schema's event types", () => {
    expect(sorted(EVENT_TYPES)).toEqual(sorted(Object.keys(schema.discriminator.mapping)));
  });

  it("declares each event's fields exactly as the schema does", () => {
    for (const [type, ref] of Object.entries<string>(schema.discriminator.mapping)) {
      const tsFields = EVENT_FIELDS[type as SearchEvent["type"]];
      expect(sorted(tsFields), type).toEqual(sorted(Object.keys(defOf(ref).properties)));
    }
  });

  it("declares each nested model's fields exactly as the schema does", () => {
    for (const [name, tsFields] of Object.entries(MODEL_FIELDS)) {
      expect(schema.$defs[name], name).toBeDefined();
      expect(sorted(tsFields), name).toEqual(sorted(Object.keys(schema.$defs[name].properties)));
    }
  });

  it("matches the schema's enumerations", () => {
    expect(sorted(STOP_REASONS)).toEqual(sorted(schema.$defs.StopReason.enum));
    expect(sorted(PHASES)).toEqual(sorted(schema.$defs.PhaseStarted.properties.phase.enum));
    expect(sorted(MODES)).toEqual(sorted(schema.$defs.SearchStarted.properties.mode.enum));
    // Action is nullable in PhaseSummary, so find the enum inside the anyOf branch
    const actionEnumDef = schema.$defs.PhaseSummary.properties.action.anyOf.find((d: Record<string, unknown>) => d.enum);
    expect(sorted(ACTIONS)).toEqual(sorted(actionEnumDef.enum));
    expect(sorted(TOOL_ERROR_KINDS)).toEqual(sorted(schema.$defs.ToolErrorInfo.properties.kind.enum));
  });
});

describe("golden SSE fixtures", () => {
  const allFrames = Object.values(fixtures).flat();

  it("are well-formed frames of known event types", () => {
    expect(allFrames.length).toBeGreaterThan(10);
    for (const frame of allFrames) {
      expect(isSearchEvent(frame.data)).toBe(true);
      expect(frame.event).toBe(frame.data.type);
      expect(frame.id).toBe(frame.data.seq);
      const allowed = EVENT_FIELDS[frame.data.type as SearchEvent["type"]];
      expect(sorted(Object.keys(frame.data))).toEqual(sorted(allowed));
    }
  });

  it("end each stream with exactly one terminal event", () => {
    for (const [name, frames] of Object.entries(fixtures)) {
      const terminal = frames.filter((f) => f.event === "search_finished" || f.event === "search_failed");
      expect(terminal, name).toHaveLength(1);
      expect(frames.at(-1), name).toBe(terminal[0]);
    }
  });

  it("carry the lean result: trace and hit content only when requested", () => {
    const lean = fixtures.retrieval!.at(-1)!.data.result as SearchResult;
    expect(lean.trace).toBeUndefined();
    expect(lean.hits.length).toBeGreaterThan(0);
    for (const h of lean.hits) expect(h.hit.content).toBeUndefined();
    const full = fixtures.with_content!.at(-1)!.data.result as SearchResult;
    expect(full.trace?.events.length).toBeGreaterThan(0);
    for (const h of full.hits) expect(h.hit.content?.length).toBeGreaterThan(0);
    const snapshot = fixtures.with_content!.find((f) => f.event === "results_updated")!.data.hits as HitSummary[];
    for (const s of snapshot) expect(s.content?.length).toBeGreaterThan(0);
  });

  it("echo back questions with null image data and uri", () => {
    const image = fixtures.image!;
    const started = image.find((f) => f.event === "search_started")!.data;
    const question = started.question as Query;
    const imagePartStarted = question.content.find((p) => (p as unknown as Record<string, unknown>).kind === "image") as ImagePart;
    expect(imagePartStarted).toBeDefined();
    expect(imagePartStarted.kind).toBe("image");
    expect(imagePartStarted.data).toBeNull();
    expect(imagePartStarted.uri).toBeNull();
    expect(typeof imagePartStarted.mime).toBe("string");

    const finished = image.find((f) => f.event === "search_finished")!.data;
    const resultQuestion = (finished.result as SearchResult).question;
    const imagePartFinished = resultQuestion.content.find((p) => (p as unknown as Record<string, unknown>).kind === "image") as ImagePart;
    expect(imagePartFinished).toBeDefined();
    expect(imagePartFinished.kind).toBe("image");
    expect(imagePartFinished.data).toBeNull();
    expect(imagePartFinished.uri).toBeNull();
    expect(typeof imagePartFinished.mime).toBe("string");
  });
});

describe("schema required-field sets", () => {
  it("SearchResult.required excludes trace and only includes the five core fields", () => {
    expect(sorted(schema.$defs.SearchResult.required as string[])).toEqual(sorted(["hits", "mode", "question", "stop_reason", "usage"]));
    expect(schema.$defs.SearchResult.required).not.toContain("trace");
  });

  it("Hit.required excludes content and only includes the two core fields", () => {
    expect(sorted(schema.$defs.Hit.required as string[])).toEqual(sorted(["doc_id", "source"]));
    expect(schema.$defs.Hit.required).not.toContain("content");
  });
});
