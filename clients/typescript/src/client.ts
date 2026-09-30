import { AgenticSearchError } from "./errors.js";
import {
  type Budget,
  type Mode,
  type SearchEvent,
  type SearchResult,
  isSearchEvent,
  isTerminal,
} from "./events.js";
import { parseSse } from "./sse.js";

export interface ClientOptions {
  /** Service root, e.g. `https://search.example.com`. */
  baseUrl: string;
  /** Sent as `Authorization: Bearer <apiKey>`. */
  apiKey?: string;
  /** Custom fetch (tests, proxies, older runtimes). Defaults to the global `fetch`. */
  fetch?: typeof fetch;
  /**
   * Extra headers on every request. The client's own `Accept`, `Content-Type` (on requests with a
   * body) and `Authorization` (when `apiKey` is set) take precedence over these.
   */
  headers?: Record<string, string>;
}

export interface ImageInput {
  /** Standard or URL-safe base64 image bytes. */
  data: string;
  mime?: string;
}

export interface SearchRequest {
  profile?: string;
  question: string;
  images?: ImageInput[];
  sources?: string[];
  mode?: Mode;
  top_k?: number;
  budget?: Partial<Budget>;
  include_content?: boolean;
  include_trace?: boolean;
}

export interface StreamRequest extends SearchRequest {
  snapshot_k?: number;
}

export interface CallOptions {
  signal?: AbortSignal;
}

export interface StreamOptions extends CallOptions {
  /** Called for event types this client does not know (a newer server); they are skipped. */
  onUnknownEvent?: (type: string, data: unknown) => void;
}

export interface SourceInfo {
  name: string;
  backend_type: string;
  capabilities: string[];
  collections: string[];
  description: string | null;
}

export interface ProfileInfo {
  name: string;
  default: boolean;
  available: boolean;
  error: string | null;
  mode: Mode;
  budget: Budget;
  limits: { max_budget: Partial<Record<keyof Budget, number>>; allow_include_content: boolean };
  sources: SourceInfo[];
  setup_errors: Record<string, string>;
}

export interface Health {
  status: "ok" | "degraded";
  version: string;
}

/** Links an optional caller signal to a controller we own, so we can also abort internally. */
function linkedController(signal?: AbortSignal): { controller: AbortController; unlink: () => void } {
  const controller = new AbortController();
  if (!signal) return { controller, unlink: () => undefined };
  if (signal.aborted) {
    controller.abort(signal.reason);
    return { controller, unlink: () => undefined };
  }
  const onAbort = () => controller.abort(signal.reason);
  signal.addEventListener("abort", onAbort, { once: true });
  return { controller, unlink: () => signal.removeEventListener("abort", onAbort) };
}

export class AgenticSearchClient {
  private readonly baseUrl: string;
  private readonly apiKey: string | undefined;
  private readonly fetchImpl: typeof fetch;
  private readonly headers: Record<string, string>;

  constructor(options: ClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.apiKey = options.apiKey;
    this.headers = options.headers ?? {};
    const f = options.fetch ?? globalThis.fetch;
    if (typeof f !== "function") {
      throw new TypeError("no fetch implementation available; pass `fetch` in the options");
    }
    this.fetchImpl = f.bind(globalThis);
  }

  health(options: CallOptions = {}): Promise<Health> {
    return this.json<Health>("GET", "/healthz", undefined, options.signal);
  }

  async profiles(options: CallOptions = {}): Promise<ProfileInfo[]> {
    const body = await this.json<{ profiles: ProfileInfo[] }>("GET", "/v1/profiles", undefined,
      options.signal);
    return body.profiles;
  }

  search(request: SearchRequest, options: CallOptions = {}): Promise<SearchResult> {
    return this.json<SearchResult>("POST", "/v1/search", request, options.signal);
  }

  /**
   * Stream one search's events. The last event is `search_finished` or `search_failed`
   * (a failed search is an event, not an exception). Breaking out of the loop, or aborting
   * `signal`, closes the connection and the service cancels the search.
   */
  async *stream(request: StreamRequest, options: StreamOptions = {}): AsyncGenerator<SearchEvent, void, undefined> {
    const { controller, unlink } = linkedController(options.signal);
    try {
      const response = await this.request("POST", "/v1/search/stream", request, controller.signal,
        { Accept: "text/event-stream" });
      if (!response.body) throw new AgenticSearchError("response has no body", response.status);
      for await (const message of parseSse(response.body)) {
        controller.signal.throwIfAborted();
        let data: unknown;
        try {
          data = JSON.parse(message.data);
        } catch {
          throw new AgenticSearchError("malformed event data", response.status, message.data);
        }
        if (data === null || typeof data !== "object" || !isSearchEvent(data as { type?: unknown })) {
          options.onUnknownEvent?.(message.event, data);
          continue;
        }
        const event = data as SearchEvent;
        yield event;
        if (isTerminal(event)) return;
      }
      controller.signal.throwIfAborted();
      throw new AgenticSearchError("stream ended before search_finished or search_failed", 0);
    } finally {
      unlink();
      controller.abort();
    }
  }

  private async json<T>(method: string, path: string, body: unknown, signal?: AbortSignal): Promise<T> {
    const response = await this.request(method, path, body, signal, { Accept: "application/json" });
    const text = await response.text();
    try {
      return JSON.parse(text) as T;
    } catch {
      throw new AgenticSearchError("response is not JSON", response.status, text);
    }
  }

  private async request(method: string, path: string, body: unknown, signal: AbortSignal | undefined,
    extra: Record<string, string>): Promise<Response> {
    const headers = new Headers(this.headers);
    for (const [name, value] of Object.entries(extra)) headers.set(name, value);
    if (body !== undefined) headers.set("Content-Type", "application/json");
    if (this.apiKey) headers.set("Authorization", `Bearer ${this.apiKey}`);
    const init: RequestInit = { method, headers };
    if (body !== undefined) init.body = JSON.stringify(body);
    if (signal) init.signal = signal;
    const response = await this.fetchImpl(`${this.baseUrl}${path}`, init);
    if (!response.ok) {
      const text = await response.text();
      const { summary, detail } = describeErrorBody(text, response);
      throw new AgenticSearchError(`${method} ${path} failed with ${response.status}: ${summary}`,
        response.status, detail);
    }
    return response;
  }
}

/**
 * The message summary and `detail` for a non-2xx body. A JSON `detail` string is used as is; a
 * FastAPI 422 `detail` array becomes `loc: msg` pairs; anything else falls back to the status
 * text, plus a short excerpt of a non-JSON body. `detail` keeps the full parsed value or raw text.
 */
function describeErrorBody(text: string, response: Response): { summary: string; detail: unknown } {
  const status = response.statusText || `HTTP ${response.status}`;
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    const excerpt = text.replace(/\s+/g, " ").trim().slice(0, 200);
    return { summary: excerpt ? `${status} — ${excerpt}` : status, detail: text };
  }
  const hasDetail = parsed !== null && typeof parsed === "object" && "detail" in parsed
    && (parsed as { detail: unknown }).detail != null;
  const detail = hasDetail ? (parsed as { detail: unknown }).detail : parsed;
  if (!hasDetail) return { summary: status, detail };
  if (typeof detail === "string" && detail) return { summary: detail, detail };
  if (Array.isArray(detail) && detail.length > 0) {
    const summary = detail.map((d: unknown) => {
      if (d === null || typeof d !== "object") return String(d);
      const { loc, msg } = d as { loc?: unknown; msg?: unknown };
      const text = typeof msg === "string" ? msg : JSON.stringify(d);
      return Array.isArray(loc) ? `${loc.join(".")}: ${text}` : text;
    }).join("; ");
    return { summary, detail };
  }
  return { summary: status, detail };
}
