/**
 * Search events, schema version 1. Mirrors `agentic_search.events` (Python) and the JSON Schema
 * the service exports to `schema/search-events.v1.schema.json`; `test/contract.test.ts` fails if
 * the two drift apart. Unknown event types from a newer server are skipped by the client.
 */

export const SCHEMA_VERSION = 1;

export type Phase = "plan" | "query" | "judge" | "decide" | "delegate";
export type Mode = "retrieval" | "harness" | "model";
export type Action = "continue" | "refine" | "broaden" | "switch_source" | "stop";
export type ToolErrorKind = "validation" | "backend" | "timeout" | "embedder" | "policy";
export type StopReason =
  | "controller_stop"
  | "no_plan"
  | "single_pass"
  | "delegate_done"
  | "budget_turns"
  | "budget_tool_calls"
  | "budget_tokens"
  | "budget_cost"
  | "budget_time";

export interface TextPart {
  kind: "text";
  text: string;
}

/** `data` is standard base64; `uri` is set instead for images held by reference. */
export interface ImagePart {
  kind: "image";
  uri: string | null;
  data: string | null;
  mime: string;
}

export interface StructuredPart {
  kind: "structured";
  data: Record<string, unknown>;
}

export type ContentPart = TextPart | ImagePart | StructuredPart;

export interface Query {
  content: ContentPart[];
}

export interface Budget {
  max_turns: number;
  max_tool_calls: number;
  max_tokens: number | null;
  max_cost_usd: number | null;
  max_seconds: number | null;
}

export interface Usage {
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  tool_calls: number;
  turns: number;
}

export interface PhaseSummary {
  n_calls: number | null;
  n_hits: number | null;
  n_new: number | null;
  n_errors: number | null;
  n_judged: number | null;
  n_relevant: number | null;
  action: Action | null;
  confidence: number | null;
  n_ranked: number | null;
  note: string | null;
  error: string | null;
}

export interface ToolErrorInfo {
  kind: ToolErrorKind;
  message: string;
  source: string | null;
}

export interface HitSummary {
  key: string;
  source: string;
  doc_id: string;
  title: string | null;
  snippet: string;
  score: number;
  p_relevant: number | null;
  judged: boolean;
  first_turn: number;
  content: ContentPart[] | null;
}

export interface OpRef {
  turn: number;
  call_id: string;
  rank: number;
}

/** A document hit. `content` is absent unless the request set `include_content`. */
export interface Hit {
  doc_id: string;
  source: string;
  content?: ContentPart[];
  metadata: Record<string, unknown>;
  raw_score: number | null;
  provenance: OpRef[];
}

export interface RankedHit {
  hit: Hit;
  score: number;
  p_relevant: number | null;
  rationale: string | null;
  judged: boolean;
}

export interface TraceEvent {
  type: string;
  turn: number;
  at_ms: number;
  duration_ms: number | null;
  data: Record<string, unknown>;
}

export interface Trace {
  events: TraceEvent[];
}

/** The service's lean result. `trace` is present only if the request set `include_trace`. */
export interface SearchResult {
  question: Query;
  hits: RankedHit[];
  stop_reason: StopReason;
  usage: Usage;
  mode: string;
  trace?: Trace;
}

export interface EventBase {
  schema_version: number;
  search_id: string;
  seq: number;
  turn: number;
  at_ms: number;
}

export interface SearchStarted extends EventBase {
  type: "search_started";
  question: Query;
  mode: Mode;
  sources: string[];
  budget: Budget;
  setup_errors: Record<string, string>;
}

export interface PhaseStarted extends EventBase {
  type: "phase_started";
  phase: Phase;
}

export interface PhaseFinished extends EventBase {
  type: "phase_finished";
  phase: Phase;
  duration_ms: number;
  summary: PhaseSummary;
}

export interface ToolCallStarted extends EventBase {
  type: "tool_call_started";
  call_id: string;
  source: string | null;
  name: string;
  arguments: Record<string, unknown>;
}

export interface ToolCallFinished extends EventBase {
  type: "tool_call_finished";
  call_id: string;
  n_hits: number;
  duration_ms: number;
  error: ToolErrorInfo | null;
}

export interface ResultsUpdated extends EventBase {
  type: "results_updated";
  hits: HitSummary[];
  pool_size: number;
  n_relevant: number;
}

export interface UsageUpdated extends EventBase {
  type: "usage_updated";
  usage: Usage;
}

export interface SearchFinished extends EventBase {
  type: "search_finished";
  result: SearchResult;
  stop_reason: StopReason;
}

export interface SearchFailed extends EventBase {
  type: "search_failed";
  error_type: string;
  message: string;
}

export type SearchEvent =
  | SearchStarted
  | PhaseStarted
  | PhaseFinished
  | ToolCallStarted
  | ToolCallFinished
  | ResultsUpdated
  | UsageUpdated
  | SearchFinished
  | SearchFailed;

export type SearchEventType = SearchEvent["type"];

export const EVENT_TYPES: readonly SearchEventType[] = [
  "search_started",
  "phase_started",
  "phase_finished",
  "tool_call_started",
  "tool_call_finished",
  "results_updated",
  "usage_updated",
  "search_finished",
  "search_failed",
];

const KNOWN = new Set<string>(EVENT_TYPES);

export function isSearchEvent(value: { type?: unknown }): value is SearchEvent {
  return typeof value.type === "string" && KNOWN.has(value.type);
}

export function isTerminal(event: SearchEvent): event is SearchFinished | SearchFailed {
  return event.type === "search_finished" || event.type === "search_failed";
}

/** The candidate key the harness uses: `<source>:<doc_id>`. */
export function hitKey(hit: Pick<Hit, "source" | "doc_id">): string {
  return `${hit.source}:${hit.doc_id}`;
}
