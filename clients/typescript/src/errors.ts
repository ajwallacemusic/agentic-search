/** An HTTP-level failure from the service, or a stream that ended without a terminal event. */
export class AgenticSearchError extends Error {
  /** HTTP status; 0 when the failure was not an HTTP response (e.g. a truncated stream). */
  readonly status: number;
  /** The service's `detail` field when present, else the raw response body. */
  readonly detail: unknown;

  constructor(message: string, status: number, detail?: unknown) {
    super(message);
    this.name = "AgenticSearchError";
    this.status = status;
    this.detail = detail;
  }
}
