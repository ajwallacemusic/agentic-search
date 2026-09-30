# Streaming Search Events — Design Spec

- **Date:** 2026-09-29
- **Status:** Approved 2026-09-29; clarified during plan verification (§2 SearchStream, §3.2, §3.3, §4.6, §5, §7)
- **Scope:** Streaming sub-project 1 of 3 (library core). Later specs: (2) standalone search
  service with SSE, startup config profiles and per-request overrides; (3) TypeScript client.

## 1. Purpose

Let a project that embeds `agentic-search` show a search's progress live (planning, querying
each source, judging, deciding) and a result list that grows turn by turn, then the final
result. The event format defined here is the contract the service (spec 2) serialises to SSE and
the TypeScript client (spec 3) mirrors.

### Goals

1. `Harness.stream(...)` returns an async iterator of typed, JSON-serialisable events scoped to
   one search.
2. Phase-level progress with start and finish events, per-backend-call events, live top-k
   snapshots, running usage, and exactly one terminal event.
3. `Harness.search(...)` keeps its signature, return type and exception behaviour, and is
   implemented on top of `stream()` (one code path).
4. Consumers can cancel a search by abandoning the iterator.
5. A versioned event schema that internal refactors do not break.

### Non-goals

- The HTTP/SSE service, config profiles, auth, multi-tenancy (spec 2).
- The TypeScript client (spec 3).
- Token-by-token streaming of model output.
- Replacing `Trace` or `Hooks.on_trace_event`; both stay as they are.

## 2. Public API

```python
class Harness:
    def stream(self, question: str | Query, *, sources: list[str] | None = None,
               top_k: int = 20, mode: Mode | None = None, budget: Budget | None = None,
               snapshot_k: int = 10, include_content: bool = False
               ) -> SearchStream: ...

    async def search(self, question: str | Query, *, sources: list[str] | None = None,
                     top_k: int = 20, mode: Mode | None = None,
                     budget: Budget | None = None) -> SearchResult: ...  # unchanged
```

`SearchStream` is an async iterator of `SearchEvent` and an async context manager; leaving the
`async with` block closes it and cancels the search.

Usage:

```python
async with load_harness("search.yaml") as h:
  async with h.stream("statin myopathy risk", sources=["pubmed"], snapshot_k=10) as events:
    async for ev in events:
        match ev:
            case PhaseStarted(phase="query"): ...
            case ResultsUpdated(hits=hits): ...
            case SearchFinished(result=result): ...
            case SearchFailed(error_type=t, message=m): ...
```

- Argument validation raises `HarnessError` from `stream()` before any event is produced:
  unknown `mode` (as `search()` does today), plus two new checks, `top_k < 1` and
  `snapshot_k < 0`. `search()` gains the `top_k < 1` check too.
  Unknown `sources` are a runtime failure (they need discovery) and produce `search_failed`.
- `snapshot_k=0` disables `results_updated` events.
- `agentic_search/__init__.py` exports `SearchEvent` and every event class.

## 3. Event model

A new module `src/agentic_search/events.py` holds frozen Pydantic v2 models. Each has a
`type: Literal[...]` field; `SearchEvent` is the discriminated union
`Annotated[Union[...], Field(discriminator="type")]`.

### 3.1 Common fields (every event)

| Field | Type | Meaning |
|---|---|---|
| `type` | literal | event type (table below) |
| `schema_version` | `int` = 1 | contract version |
| `search_id` | `str` | UUID4 hex, fixed for one search |
| `seq` | `int` | 0, 1, 2, … strictly increasing, no gaps, per search |
| `turn` | `int` | current loop turn (0-based) |
| `at_ms` | `float` | milliseconds since the search started (monotonic clock) |

### 3.2 Event types

| `type` | Class | Emitted | Payload fields |
|---|---|---|---|
| `search_started` | `SearchStarted` | first, once setup and source selection succeed | `question: Query`, `mode: str`, `sources: list[str]`, `budget: Budget`, `setup_errors: dict[str, str]` |
| `phase_started` | `PhaseStarted` | before a phase | `phase: Phase` |
| `phase_finished` | `PhaseFinished` | after that phase | `phase: Phase`, `duration_ms: float`, `summary: PhaseSummary` |
| `tool_call_started` | `ToolCallStarted` | before each backend call | `call_id: str`, `source: str \| None`, `name: str`, `arguments: dict` |
| `tool_call_finished` | `ToolCallFinished` | when each backend call completes | `call_id: str`, `n_hits: int`, `duration_ms: float`, `error: ToolErrorInfo \| None` (new-candidate counts are per turn, on `phase_finished("query")`, because de-duplication across concurrent calls is only settled once the turn's calls are pooled) |
| `results_updated` | `ResultsUpdated` | after each judge phase; after each delegate tool round | `hits: list[HitSummary]`, `pool_size: int`, `n_relevant: int` |
| `usage_updated` | `UsageUpdated` | after each turn's `results_updated` point, after each `decide` phase, after each delegate tool round, and once after the delegate finishes | `usage: Usage` (running totals) |
| `search_finished` | `SearchFinished` | last, on success | `result: SearchResult`, `stop_reason: StopReason` |
| `search_failed` | `SearchFailed` | last, on error | `error_type: str`, `message: str` |

`Phase = Literal["plan", "query", "judge", "decide", "delegate"]`.

`PhaseSummary` (all fields optional, filled per phase):
- `plan`: `n_calls`
- `query`: `n_calls`, `n_hits`, `n_new`, `n_errors`
- `judge`: `n_judged`, `n_relevant`, `error` (scrubbed message if the decider failed and
  the analyzer fell back)
- `decide`: `action` (the `Action` value), `confidence`, `error` (if the decider failed
  and the controller fell back)
- `delegate`: `n_ranked`, `note`

`ToolErrorInfo`: `kind: str`, `message: str` (scrubbed), `source: str | None` — mirrors the
existing `ToolError`.

`HitSummary`:

| Field | Type | Notes |
|---|---|---|
| `key` | `str` | candidate key (`source:doc_id`) |
| `source` | `str` | |
| `doc_id` | `str` | |
| `title` | `str \| None` | from hit metadata/title when present |
| `snippet` | `str` | `Hit.snippet(300)`, scrubbed |
| `score` | `float` | same scoring as the final ranking |
| `p_relevant` | `float \| None` | |
| `judged` | `bool` | |
| `first_turn` | `int` | turn the hit was first seen |
| `content` | `list[ContentPart] \| None` | only when `include_content=True` |

### 3.3 Ordering guarantees

- `search_started` is `seq=0`, except when setup or source selection fails: then `search_failed`
  is the only event.
- Every `phase_started` has a matching `phase_finished` for the same phase and turn unless
  the search is cancelled or fails inside it.
- Every `tool_call_started` has a matching `tool_call_finished` with the same `call_id`
  (same exception).
- Exactly one terminal event (`search_finished` or `search_failed`) ends every stream that is
  not cancelled; nothing follows it.
- Within a turn, tool-call events interleave in real completion order.

### 3.4 Per-mode event sequences

- **retrieval**: started → plan → query (tool calls) → judge → results_updated →
  usage_updated → finished.
- **harness**: retrieval's turn, then decide, repeated per turn until stop → finished.
- **model**: started → delegate phase_started → per tool round: tool calls,
  results_updated (snapshots in delegate mode are ranked by pool score, since the delegate's
  own order is only known at the end) → delegate phase_finished → judge (if an analyzer
  decider is configured) → results_updated → usage_updated → finished.

Budget stops emit no special event; the stop reason is on `search_finished`.

### 3.5 Snapshot semantics

`results_updated.hits` is the top `snapshot_k` of `state.pool.ranked(judge_weight)`, built by
the same function `_finalize` uses to produce `RankedHit`s (factored into one helper). In
retrieval and harness modes the final `results_updated` therefore equals the head of
`SearchFinished.result.hits` (keys, order, scores). In model mode the final result may reorder
by the delegate's ranking; that difference is documented, not hidden.

### 3.6 Versioning

`schema_version = 1`. Adding an optional field or a new event type keeps the version; renaming
or removing a field, changing a type, or changing ordering guarantees bumps it. Consumers must
ignore unknown event types.

## 4. Internals

### 4.1 EventEmitter

`src/agentic_search/core/emitter.py`:

```python
class EventEmitter:
    def __init__(self, search_id: str, sink: Callable[[SearchEvent], None],
                 clock: Callable[[], float] = time.monotonic): ...
    def emit(self, cls: type[E], *, turn: int, **fields: Any) -> E: ...

class NullEmitter(EventEmitter):  # sink discards; default on SearchState
```

`emit` stamps the common fields, applies `scrub_data` to string payloads (arguments, error
messages, snippets, notes), constructs the model, and calls the sink synchronously. `seq` is
assigned inside `emit`; because `emit` is synchronous on one event loop, `seq` is gap-free and
monotonic even with concurrent tool calls.

`SearchState` gains `emitter: EventEmitter = field(default_factory=NullEmitter)`, so roles
constructed in existing unit tests keep working unchanged.

### 4.2 Emission points

| Location | Events |
|---|---|
| `Harness._run_search` start | `search_started` (after setup and source selection) |
| `Planner.plan` | `phase_started/finished("plan")` |
| `Harness._run_loop` around `executor.run` | `phase_started/finished("query")` |
| `Executor._run_one` | `tool_call_started`, `tool_call_finished` (executor takes an `emitter` argument alongside `trace`) |
| `Analyzer.analyze` / `judge_keys` | `phase_started/finished("judge")` |
| `Harness` after analyze; `_DelegateRuntime.call` after each round | `results_updated` (unless `snapshot_k == 0`), `usage_updated` |
| `Controller.decide` | `phase_started/finished("decide")` |
| `Harness._run_delegate` | `phase_started/finished("delegate")` |
| `Harness._run_search` end | `search_finished` or `search_failed` |

Existing `trace.add(...)` calls are unchanged.

### 4.3 stream() and search()

```python
async def stream(self, question, **opts) -> AsyncIterator[SearchEvent]:
    self._validate(opts)                           # raises HarnessError eagerly
    queue: asyncio.Queue[SearchEvent] = asyncio.Queue()
    emitter = EventEmitter(uuid4().hex, queue.put_nowait)
    task = asyncio.create_task(self._run_search(question, emitter, **opts))
    try:
        while True:
            ev = await queue.get()
            yield ev
            if isinstance(ev, (SearchFinished, SearchFailed)):
                break
    finally:
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
```

Because `stream()` is an async generator, argument validation must run before the first
`yield`; the implementation uses a small wrapper (a regular method that validates, then returns
the async generator) so `HarnessError` is raised at call time, not at first iteration.

- `_run_search` wraps the existing setup/select/loop/finalize code in `try/except Exception`.
  On exception it emits `search_failed` (`error_type=type(exc).__name__`, scrubbed `str(exc)`)
  and stores the exception on the task result for `search()`. `CancelledError` is not caught.
- The queue is unbounded: the event count is bounded by the budget (turns × calls), and a
  bounded queue would let a slow consumer stall backend calls mid-flight.
- `search()` iterates `stream()` internally with `snapshot_k=0`, returns
  `SearchFinished.result`, and on `SearchFailed` re-raises the original exception object
  (retrieved from the task), preserving today's exception types.

### 4.4 Cancellation

Abandoning the iterator (`break`, `aclose()`, garbage collection, or cancellation of the
consuming task) runs the generator's `finally`, which cancels the search task. `CancelledError`
propagates through `asyncio.gather` in the executor and cancels in-flight backend and model
calls. No events are emitted after cancellation; the stream simply ends.

### 4.5 Concurrency

One `Harness` supports many concurrent streams: each has its own emitter, queue, state and
task. First-call `setup()` is already serialised by `Harness._setup_lock`; this spec adds a
test for concurrent first calls.

### 4.6 Security

Events go to the embedding application, not to a model, so `SourcePolicy` (which governs
model visibility) does not filter them. Secrets registered with `core.secrets` are scrubbed
from tool arguments, error messages, notes, hit titles and snippets. Two payloads are passed
through unmodified: `SearchFinished.result` is exactly the object `search()` returns (its trace
is already scrubbed; document content is not rewritten), and `HitSummary.content`, which appears
only with `include_content=True`. By default a scrubbed 300-character snippet is the most
document content an event carries.

## 5. Error handling

| Situation | Behaviour |
|---|---|
| Invalid arguments | `HarnessError` raised from `stream()` / `search()` before any event |
| Setup fails, no reachable backends, unknown sources | `search_failed` as the only event; `search()` raises the original `HarnessError` |
| Unexpected exception in the loop | `search_failed`; `search()` re-raises it |
| One backend call fails | `tool_call_finished.error`; search continues (existing degradation) |
| Judge/decider fails or times out | `phase_finished.summary.error`; fallback behaviour as today |
| `Hooks.on_trace_event` raises | unchanged from today |
| Consumer cancels | search task cancelled; no terminal event |

## 6. Testing

Unit (no services; `ScriptedDriver`, fake deciders, `FilesBackend`):

1. Event sequence per mode (retrieval, harness, model) matches §3.4; `seq` gap-free from 0;
   single `search_id`; exactly one terminal event.
2. Final `results_updated` equals the head of the final result in retrieval and harness modes.
3. Parallel tool calls: every `tool_call_started` pairs with a `tool_call_finished`; `seq`
   stays gap-free.
4. Cancellation: breaking after the first `tool_call_started` against a slow fake backend
   cancels the backend coroutine; no further events.
5. Failures: unknown source → `search_failed` and `search()` raises `HarnessError`; a failing
   backend call → `tool_call_finished.error` and the search still finishes; a failing
   decider → `phase_finished.summary.error`.
6. Scrubbing: a registered secret placed in tool arguments, a backend error message and a
   document snippet appears in no serialised event.
7. `include_content`: `HitSummary.content` is `None` by default and populated when set.
8. Two concurrent streams on one harness: distinct `search_id`s, no cross-talk; concurrent
   first calls run discovery once.
9. JSON round-trip: every event class survives `model_dump_json()` →
   `TypeAdapter(SearchEvent).validate_json()` unchanged.
10. `snapshot_k=0` emits no `results_updated`; argument validation raises before any event.
11. Regression: the existing suite passes unchanged.

Integration (contract suite, Docker-gated): stream a search against the files backend and one
Docker backend; assert it ends in `search_finished` and its result equals `search()` for the
same scripted plan.

## 7. Files

- Create: `src/agentic_search/events.py`, `src/agentic_search/core/emitter.py`,
  `src/agentic_search/core/result.py` (`RankedHit` and `SearchResult` move here from
  `core/harness.py`, which re-exports them, so `events.py` can import them without a cycle),
  `tests/test_events.py`, `tests/test_stream.py`, `tests/roles/test_role_events.py`
- Modify: `src/agentic_search/core/harness.py`, `core/state.py`, `roles/planner.py`,
  `roles/executor.py`, `roles/analyzer.py`, `roles/controller.py`,
  `src/agentic_search/__init__.py`, `README.md`, `tests/contract/test_backend_contract.py`
