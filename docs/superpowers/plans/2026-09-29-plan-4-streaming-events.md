# Streaming Search Events Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `Harness.stream()`, an async iterator of typed, JSON-serialisable events for one search (phase start/finish, per-call events, live top-k snapshots, running usage, one terminal event), and rebuild `Harness.search()` on top of it.

**Architecture:** A new `agentic_search.events` module defines frozen Pydantic event models joined in a discriminated union, `SearchEvent`. Each search gets its own `EventEmitter` (stored on `SearchState`), which stamps `search_id`/`seq`/`turn`/`at_ms`, scrubs payloads, and pushes events onto an `asyncio.Queue`. The planner, executor, analyzer and controller emit at their phase boundaries. `Harness.stream()` runs the search as a task, yields from the queue, and cancels the task if the consumer stops. `search()` drains the stream and returns the final result, re-raising the original exception on failure.

**Tech Stack:** Python ≥3.11, asyncio, Pydantic v2, pytest (`asyncio_mode = "auto"`), uv, ruff.

**Spec:** `docs/superpowers/specs/2026-09-29-streaming-events-design.md`

## Global Constraints

- `Harness.search()` keeps its signature, return type (`SearchResult`) and exception types. The whole existing suite must pass unchanged at every task boundary.
- `Trace`, `trace.add(...)` calls and `Hooks.on_trace_event` stay as they are. Events are added next to them and do not replace them.
- Every event carries `schema_version = 1`, `search_id`, `seq` (0-based, gap-free, strictly increasing per search), `turn` and `at_ms`.
- Every stream that is not cancelled ends with exactly one `search_finished` or `search_failed`, and nothing follows it.
- Registered secrets never appear in tool arguments, error messages, notes, hit titles or snippets inside events. `SearchFinished.result` and opt-in `HitSummary.content` are passed through unmodified.
- `SourcePolicy` does not filter events. Full hit content only with `include_content=True`.
- No new runtime dependencies.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Run tests with `uv run pytest -q` and lint with `uv run ruff check` from the repo root. Both must be clean at the end of every task.

## File Structure

| File | Responsibility |
|---|---|
| `src/agentic_search/core/result.py` (new) | `RankedHit`, `SearchResult`, moved from `core/harness.py` so `events.py` can import them without an import cycle. `core/harness.py` re-exports both. |
| `src/agentic_search/events.py` (new) | Event models, `PhaseSummary`, `ToolErrorInfo`, `HitSummary`, the `SearchEvent` union, `TERMINAL_EVENTS`, `SCHEMA_VERSION`. |
| `src/agentic_search/core/emitter.py` (new) | `EventEmitter` (stamps common fields, scrubs, calls the sink) and `NullEmitter`. It must not import `events.py`: `core/state.py` imports it, and `events.py` imports `core/state.py`. |
| `src/agentic_search/core/state.py` | `SearchState.emitter` field (defaults to `NullEmitter`). |
| `src/agentic_search/roles/{planner,executor,analyzer,controller}.py` | Emit `plan`, tool-call, `judge` and `decide` events. |
| `src/agentic_search/core/harness.py` | `stream()`, `SearchStream`, `search()` on top of `stream()`, the `query`/`delegate` phases, snapshots, usage events. |
| `src/agentic_search/__init__.py`, `README.md` | Public exports and a Streaming section. |
| `tests/test_events.py`, `tests/roles/test_role_events.py`, `tests/test_stream.py`, `tests/contract/test_backend_contract.py` | Tests. |

---

### Task 1: Event models, result module and emitter

**Files:**
- Create: `src/agentic_search/core/result.py`, `src/agentic_search/events.py`, `src/agentic_search/core/emitter.py`, `tests/test_events.py`
- Modify: `src/agentic_search/core/harness.py` (move two classes out), `src/agentic_search/core/state.py` (one field)

**Interfaces:**
- Produces:
  - `agentic_search.core.result.RankedHit` and `SearchResult`. These are the same classes as before; `agentic_search.core.harness` still exports both names.
  - `agentic_search.events`:
    - `SCHEMA_VERSION = 1`
    - `Phase = Literal["plan", "query", "judge", "decide", "delegate"]`
    - `PhaseSummary`, `ToolErrorInfo(kind, message, source)`, `HitSummary(key, source, doc_id, title, snippet, score, p_relevant, judged, first_turn, content)`
    - the event classes `SearchStarted`, `PhaseStarted`, `PhaseFinished`, `ToolCallStarted`, `ToolCallFinished`, `ResultsUpdated`, `UsageUpdated`, `SearchFinished`, `SearchFailed`
    - `SearchEvent` (the discriminated union on `type`) and `TERMINAL_EVENTS = (SearchFinished, SearchFailed)`
  - `agentic_search.core.emitter`:
    - `EventEmitter(search_id: str, sink: Callable[[Any], None], *, clock=time.monotonic)`
    - `.emit(cls, *, turn: int, **fields) -> event`
    - `.search_id`
    - `NullEmitter()`
  - `SearchState.emitter: EventEmitter`, which defaults to `NullEmitter()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_events.py`:

```python
import pytest
from pydantic import TypeAdapter, ValidationError

from agentic_search.core.emitter import EventEmitter, NullEmitter
from agentic_search.core.result import RankedHit, SearchResult
from agentic_search.core.secrets import register_secret
from agentic_search.core.state import SearchState, Trace, Usage
from agentic_search.core.types import Budget, Hit, Query, StopReason, TextPart
from agentic_search.events import (
    SCHEMA_VERSION,
    HitSummary,
    PhaseFinished,
    PhaseStarted,
    PhaseSummary,
    ResultsUpdated,
    SearchEvent,
    SearchFailed,
    SearchFinished,
    SearchStarted,
    ToolCallFinished,
    ToolCallStarted,
    ToolErrorInfo,
    UsageUpdated,
)

ADAPTER = TypeAdapter(SearchEvent)
COMMON = {"search_id": "s1", "seq": 0, "turn": 0, "at_ms": 1.5}


def one_of_each():
    hit = Hit(doc_id="d1", source="docs", content=[TextPart(text="aspirin")])
    result = SearchResult(question=Query.of("q"), hits=[RankedHit(hit=hit, score=0.5)],
                          stop_reason=StopReason.SINGLE_PASS, usage=Usage(turns=1),
                          trace=Trace(), mode="retrieval")
    summary = HitSummary(key="docs:d1", source="docs", doc_id="d1", snippet="aspirin", score=0.5,
                         first_turn=0, content=[TextPart(text="aspirin")])
    return [
        SearchStarted(**COMMON, question=Query.of("q"), mode="harness", sources=["docs"],
                      budget=Budget()),
        PhaseStarted(**COMMON, phase="plan"),
        PhaseFinished(**COMMON, phase="query", duration_ms=2.0,
                      summary=PhaseSummary(n_calls=1, n_hits=3, n_new=2, n_errors=0)),
        ToolCallStarted(**COMMON, call_id="t0.c0", source="docs", name="lexical_search",
                        arguments={"text": "x"}),
        ToolCallFinished(**COMMON, call_id="t0.c0", n_hits=0, duration_ms=1.0,
                         error=ToolErrorInfo(kind="backend", message="down", source="docs")),
        ResultsUpdated(**COMMON, hits=[summary], pool_size=1, n_relevant=0),
        UsageUpdated(**COMMON, usage=Usage(input_tokens=5, tool_calls=1)),
        SearchFinished(**COMMON, result=result, stop_reason=StopReason.SINGLE_PASS),
        SearchFailed(**COMMON, error_type="HarnessError", message="boom"),
    ]


@pytest.mark.parametrize("event", one_of_each(), ids=lambda e: e.type)
def test_json_round_trip(event):
    back = ADAPTER.validate_json(event.model_dump_json())
    assert type(back) is type(event)
    assert back.model_dump() == event.model_dump()
    assert back.schema_version == SCHEMA_VERSION == 1


def test_events_are_frozen_and_types_unique():
    ev = PhaseStarted(**COMMON, phase="plan")
    with pytest.raises(ValidationError):
        ev.seq = 5
    assert len({e.type for e in one_of_each()}) == 9


def test_unknown_phase_rejected():
    with pytest.raises(ValidationError):
        PhaseStarted(**COMMON, phase="dance")


def test_emitter_stamps_seq_turn_and_time():
    got = []
    clock = iter([100.0, 100.25, 100.5])
    em = EventEmitter("abc", got.append, clock=lambda: next(clock))
    a = em.emit(PhaseStarted, turn=0, phase="plan")
    b = em.emit(PhaseStarted, turn=2, phase="query")
    assert got == [a, b]
    assert (a.search_id, a.seq, a.turn, a.at_ms) == ("abc", 0, 0, 250.0)
    assert (b.seq, b.turn, b.at_ms) == (1, 2, 500.0)


def test_emitter_scrubs_string_payloads():
    register_secret("sk-live-9999")
    got = []
    em = EventEmitter("abc", got.append)
    em.emit(ToolCallStarted, turn=0, call_id="c", name="lexical_search",
            arguments={"text": "key sk-live-9999", "nested": ["sk-live-9999"]})
    em.emit(SearchFailed, turn=0, error_type="X", message="auth sk-live-9999 rejected")
    dumped = "".join(e.model_dump_json() for e in got)
    assert "sk-live-9999" not in dumped and "***" in dumped


def test_null_emitter_is_state_default():
    s = SearchState(question=Query.of("q"), manifests={}, budget=Budget())
    assert isinstance(s.emitter, NullEmitter)
    ev = s.emitter.emit(PhaseStarted, turn=0, phase="plan")
    assert ev.phase == "plan"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -q tests/test_events.py`
Expected: collection error `ModuleNotFoundError: No module named 'agentic_search.core.emitter'`.

- [ ] **Step 3: Create `src/agentic_search/core/result.py`**

```python
"""The final result of one search."""

from __future__ import annotations

from pydantic import BaseModel

from agentic_search.core.state import Trace, Usage
from agentic_search.core.types import Hit, Query, StopReason


class RankedHit(BaseModel):
    hit: Hit
    score: float
    p_relevant: float | None = None
    rationale: str | None = None
    judged: bool = False


class SearchResult(BaseModel):
    question: Query
    hits: list[RankedHit]
    stop_reason: StopReason
    usage: Usage
    trace: Trace
    mode: str

    def keys(self) -> list[str]:
        return [h.hit.key for h in self.hits]
```

- [ ] **Step 4: Create `src/agentic_search/events.py`**

```python
"""Typed, JSON-serialisable events for one streamed search (schema version 1).

`SearchEvent` is a discriminated union on `type`. Consumers must ignore event types they do not
know: adding an optional field or a new event type keeps `schema_version`; renaming or removing
a field, changing a type or changing ordering guarantees bumps it."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from agentic_search.core.result import SearchResult
from agentic_search.core.state import Usage
from agentic_search.core.types import Budget, Content, Query, StopReason

SCHEMA_VERSION = 1
Phase = Literal["plan", "query", "judge", "decide", "delegate"]


class _Event(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = SCHEMA_VERSION
    search_id: str
    seq: int
    turn: int
    at_ms: float


class PhaseSummary(BaseModel):
    """Per-phase counts; only the fields relevant to the phase are set."""

    model_config = ConfigDict(frozen=True)

    n_calls: int | None = None
    n_hits: int | None = None
    n_new: int | None = None
    n_errors: int | None = None
    n_judged: int | None = None
    n_relevant: int | None = None
    action: str | None = None
    confidence: float | None = None
    n_ranked: int | None = None
    note: str | None = None
    error: str | None = None


class ToolErrorInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: str
    message: str
    source: str | None = None


class HitSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    source: str
    doc_id: str
    title: str | None = None
    snippet: str
    score: float
    p_relevant: float | None = None
    judged: bool = False
    first_turn: int
    content: list[Content] | None = None


class SearchStarted(_Event):
    type: Literal["search_started"] = "search_started"
    question: Query
    mode: str
    sources: list[str]
    budget: Budget
    setup_errors: dict[str, str] = Field(default_factory=dict)


class PhaseStarted(_Event):
    type: Literal["phase_started"] = "phase_started"
    phase: Phase


class PhaseFinished(_Event):
    type: Literal["phase_finished"] = "phase_finished"
    phase: Phase
    duration_ms: float
    summary: PhaseSummary = Field(default_factory=PhaseSummary)


class ToolCallStarted(_Event):
    type: Literal["tool_call_started"] = "tool_call_started"
    call_id: str
    source: str | None = None
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCallFinished(_Event):
    type: Literal["tool_call_finished"] = "tool_call_finished"
    call_id: str
    n_hits: int
    duration_ms: float
    error: ToolErrorInfo | None = None


class ResultsUpdated(_Event):
    type: Literal["results_updated"] = "results_updated"
    hits: list[HitSummary]
    pool_size: int
    n_relevant: int


class UsageUpdated(_Event):
    type: Literal["usage_updated"] = "usage_updated"
    usage: Usage


class SearchFinished(_Event):
    type: Literal["search_finished"] = "search_finished"
    result: SearchResult
    stop_reason: StopReason


class SearchFailed(_Event):
    type: Literal["search_failed"] = "search_failed"
    error_type: str
    message: str


SearchEvent = Annotated[
    Union[SearchStarted, PhaseStarted, PhaseFinished, ToolCallStarted, ToolCallFinished,
          ResultsUpdated, UsageUpdated, SearchFinished, SearchFailed],
    Field(discriminator="type"),
]
TERMINAL_EVENTS = (SearchFinished, SearchFailed)
```

- [ ] **Step 5: Create `src/agentic_search/core/emitter.py`**

```python
"""Per-search event emitter: stamps the common fields, scrubs payloads, hands events to a sink."""

from __future__ import annotations

import time
from typing import Any, Callable, TypeVar

from agentic_search.core.secrets import scrub_data

E = TypeVar("E")


class EventEmitter:
    """`emit` is synchronous, so `seq` stays gap-free and ordered even with concurrent tool calls
    on one event loop."""

    def __init__(self, search_id: str, sink: Callable[[Any], None], *,
                 clock: Callable[[], float] = time.monotonic):
        self.search_id = search_id
        self._sink = sink
        self._clock = clock
        self._t0 = clock()
        self._seq = 0

    def emit(self, cls: type[E], *, turn: int, **fields: Any) -> E:
        event = cls(search_id=self.search_id, seq=self._seq, turn=turn,  # type: ignore[call-arg]
                    at_ms=(self._clock() - self._t0) * 1000, **scrub_data(fields))
        self._seq += 1
        self._sink(event)
        return event


class NullEmitter(EventEmitter):
    """Discards events; the default on SearchState so roles work without a stream."""

    def __init__(self) -> None:
        super().__init__("", lambda _event: None)
```

- [ ] **Step 6: Move the result classes out of `core/harness.py`**

In `src/agentic_search/core/harness.py`:
1. Delete the whole `class RankedHit(BaseModel):` and `class SearchResult(BaseModel):` definitions, including the `keys()` method. Everything between `class HarnessSettings` and `class _DelegateRuntime:` except `HarnessSettings` itself goes.
2. Insert this line directly **before** `from agentic_search.core.secrets import scrub` (ruff's isort order):
   ```python
   from agentic_search.core.result import RankedHit, SearchResult
   ```
3. Change `from agentic_search.core.state import Candidate, SearchState, Trace, Usage` to `from agentic_search.core.state import Candidate, SearchState`.
4. Change `from agentic_search.core.types import Budget, Hit, Manifest, ModelUsage, Query, StopReason` to `from agentic_search.core.types import Budget, Manifest, ModelUsage, Query, StopReason`.

`from agentic_search.core.harness import RankedHit, SearchResult` and `from agentic_search import SearchResult` keep working through the re-import.

- [ ] **Step 7: Add the emitter field to `SearchState`**

In `src/agentic_search/core/state.py`, add the import directly above `from agentic_search.core.secrets import scrub_data`:
```python
from agentic_search.core.emitter import EventEmitter, NullEmitter
```
Then add this as the last field of the `SearchState` dataclass, after `started`:
```python
    emitter: EventEmitter = field(default_factory=NullEmitter)
```

- [ ] **Step 8: Run the tests**

Run: `uv run pytest -q tests/test_events.py`
Expected: `14 passed`.

Run: `uv run pytest -q && uv run ruff check`
Expected: `357 passed, 5 skipped, 79 deselected` and `All checks passed!`.

- [ ] **Step 9: Commit**

```bash
git add src/agentic_search/core/result.py src/agentic_search/events.py src/agentic_search/core/emitter.py src/agentic_search/core/harness.py src/agentic_search/core/state.py tests/test_events.py
git commit -m "feat(events): typed search event models and per-search emitter

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Roles emit phase and tool-call events

**Files:**
- Modify: `src/agentic_search/roles/planner.py`, `src/agentic_search/roles/executor.py`, `src/agentic_search/roles/analyzer.py`, `src/agentic_search/roles/controller.py`
- Test: `tests/roles/test_role_events.py`

**Interfaces:**
- Consumes (Task 1): `SearchState.emitter`, `EventEmitter.emit`, `PhaseStarted`, `PhaseFinished`, `PhaseSummary`, `ToolCallStarted`, `ToolCallFinished`, `ToolErrorInfo`.
- Produces:
  - **`Planner.plan`:** brackets its work with `phase_started("plan")` and `phase_finished("plan", summary.n_calls)`.
  - **`Executor.run`:**
    - Gains the keyword argument `emitter: EventEmitter | None = None`; `None` means no events.
    - For each call it emits `tool_call_started` before the call and `tool_call_finished` as soon as that call completes. `tool_call_finished` carries `n_hits`, `duration_ms` and an `error: ToolErrorInfo | None` with a scrubbed message.
    - `source` is `arguments["source"]` when it is a string.
  - **`Analyzer.judge_keys`:**
    - Always brackets its work with a `judge` phase, even when there is nothing to judge or no decider.
    - The summary reports `n_judged`, `n_relevant` and the scrubbed last batch `error`.
    - The judging logic moves unchanged into `Analyzer._judge`.
  - **`Controller.decide`:**
    - Brackets its work with a `decide` phase.
    - The summary reports `action`, `confidence`, and the scrubbed decider `error` when it fell back to the heuristic.
  - **Unchanged trace messages:** the `judge_error` and `decision_error` messages keep the `"<Type>: <message>"` shape. They are now scrubbed first and then cut to 300 characters.

- [ ] **Step 1: Write the failing test**

Create `tests/roles/test_role_events.py`:

```python
"""Each role brackets its work with phase events on the state's emitter."""

from agentic_search.core.emitter import EventEmitter
from agentic_search.core.secrets import register_secret
from agentic_search.core.state import SearchState
from agentic_search.core.types import Budget, Hit, Query, TextPart
from agentic_search.embedders.base import EmbedderRegistry
from agentic_search.events import (
    ToolCallFinished,
    ToolCallStarted,
)
from agentic_search.models.base import Action
from agentic_search.roles.analyzer import Analyzer
from agentic_search.roles.controller import Controller
from agentic_search.roles.executor import ExecResult, Executor
from agentic_search.roles.planner import Planner
from agentic_search.testing import (
    FailingDecider,
    KeywordJudge,
    ScriptedController,
    ScriptedDriver,
    call,
)


def recorded_state():
    events = []
    s = SearchState(question=Query.of("fever"), manifests={}, budget=Budget(),
                    emitter=EventEmitter("sid", events.append))
    return s, events


def kinds(events):
    return [(type(e).__name__, getattr(e, "phase", None)) for e in events]


async def test_planner_emits_plan_phase():
    s, events = recorded_state()
    driver = ScriptedDriver([[call("lexical_search", source="docs", text="a"),
                              call("lexical_search", source="docs", text="b")]])
    await Planner(driver, max_calls_per_turn=1).plan(s, [])
    assert kinds(events) == [("PhaseStarted", "plan"), ("PhaseFinished", "plan")]
    assert events[1].summary.n_calls == 1 and events[1].duration_ms >= 0


def _pool_state():
    s, events = recorded_state()
    hits = [Hit(doc_id="1", source="s", content=[TextPart(text="high fever")]),
            Hit(doc_id="2", source="s", content=[TextPart(text="castles")])]
    new = s.pool.add(hits, turn=0, call_id="c")
    return s, events, new


async def test_analyzer_emits_judge_phase_with_counts():
    s, events, new = _pool_state()
    await Analyzer(KeywordJudge(["fever"])).judge_keys(s, new)
    assert kinds(events) == [("PhaseStarted", "judge"), ("PhaseFinished", "judge")]
    summary = events[1].summary
    assert (summary.n_judged, summary.n_relevant, summary.error) == (2, 1, None)


async def test_analyzer_judge_phase_reports_failure_and_no_judge():
    s, events, new = _pool_state()
    await Analyzer(FailingDecider()).judge_keys(s, new)
    assert events[1].summary.n_judged == 0
    assert events[1].summary.error == "RuntimeError: judge exploded"
    s2, events2, new2 = _pool_state()
    await Analyzer(None).judge_keys(s2, new2)
    assert kinds(events2) == [("PhaseStarted", "judge"), ("PhaseFinished", "judge")]
    assert events2[1].summary.n_judged == 0


async def test_controller_emits_decide_phase():
    s, events = recorded_state()
    await Controller(ScriptedController([Action.BROADEN])).decide(s)
    assert kinds(events) == [("PhaseStarted", "decide"), ("PhaseFinished", "decide")]
    assert events[1].summary.action == "broaden" and events[1].summary.error is None
    s2, events2 = recorded_state()
    await Controller(FailingDecider()).decide(s2)
    assert events2[1].summary.error == "RuntimeError: decide exploded"
    assert events2[1].summary.action == "continue"  # heuristic fallback, no history yet


async def test_executor_pairs_tool_call_events(docs_backend):
    manifests = {"docs": await docs_backend.discover()}
    ex = Executor({"docs": docs_backend}, manifests, EmbedderRegistry([docs_backend.embedder]))
    s, events = recorded_state()
    register_secret("sk-arg-4242")
    calls = [call("lexical_search", id="t0.c0", source="docs", text="headache sk-arg-4242"),
             call("lexical_search", id="t0.c1", source="nope", text="x")]
    res = await ex.run(calls, question=s.question, turn=0, pool=s.pool, trace=s.trace,
                       emitter=s.emitter)
    started = [e for e in events if isinstance(e, ToolCallStarted)]
    finished = {e.call_id: e for e in events if isinstance(e, ToolCallFinished)}
    assert [e.call_id for e in started] == ["t0.c0", "t0.c1"]
    assert started[0].source == "docs" and "sk-arg-4242" not in started[0].model_dump_json()
    assert set(finished) == {"t0.c0", "t0.c1"}
    assert finished["t0.c0"].error is None and finished["t0.c0"].n_hits == len(res.hits_per_call["t0.c0"])
    assert finished["t0.c1"].error.kind == "validation" and finished["t0.c1"].n_hits == 0
    assert [e.seq for e in events] == list(range(len(events)))


async def test_executor_without_emitter_still_runs(docs_backend):
    manifests = {"docs": await docs_backend.discover()}
    ex = Executor({"docs": docs_backend}, manifests, EmbedderRegistry([docs_backend.embedder]))
    s, _ = recorded_state()
    res = await ex.run([call("lexical_search", source="docs", text="headache")],
                       question=s.question, turn=0, pool=s.pool, trace=s.trace)
    assert isinstance(res, ExecResult) and res.new_keys
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -q tests/roles/test_role_events.py`
Expected: `5 failed, 1 passed`.
- The planner, analyzer and controller tests fail because no events are recorded.
- `test_executor_pairs_tool_call_events` fails with `TypeError: Executor.run() got an unexpected keyword argument 'emitter'`.
- `test_executor_without_emitter_still_runs` already passes. It guards backward compatibility.

- [ ] **Step 3: Replace `src/agentic_search/roles/planner.py`**

```python
"""Builds the planner's view of the search and asks the driver for the next batch of calls."""

from __future__ import annotations

import time

from agentic_search.core.hooks import Hooks
from agentic_search.core.state import SearchState
from agentic_search.core.types import Manifest
from agentic_search.events import PhaseFinished, PhaseStarted, PhaseSummary
from agentic_search.models.base import Driver, PlannerView, PlanResult, ToolSpec


def render_manifests(manifests: dict[str, Manifest]) -> str:
    return "\n\n".join(m.summary() for _, m in sorted(manifests.items()))


class Planner:
    def __init__(self, driver: Driver, *, hooks: Hooks | None = None, max_calls_per_turn: int = 8):
        self.driver = driver
        self.hooks = hooks or Hooks()
        self.max_calls = max_calls_per_turn

    async def plan(self, state: SearchState, tools: list[ToolSpec]) -> PlanResult:
        view = PlannerView(question=state.question, turn=state.turn,
                           manifest_summary=render_manifests(state.manifests), digest=state.digest,
                           directive=state.last_decision, errors=state.last_errors,
                           max_calls=self.max_calls)
        state.emitter.emit(PhaseStarted, turn=state.turn, phase="plan")
        t_phase = time.perf_counter()
        view = await self.hooks.before_model_call(self.driver.id, view)
        t0 = time.perf_counter()
        result = await self.driver.plan(view, tools)
        calls = [c.model_copy(update={"id": f"t{state.turn}.c{i}"})
                 for i, c in enumerate(result.calls[: self.max_calls])]
        state.trace.add("plan", state.turn, duration_ms=(time.perf_counter() - t0) * 1000,
                        driver=self.driver.id, n_calls=len(calls),
                        dropped=max(0, len(result.calls) - self.max_calls), note=result.note)
        state.emitter.emit(PhaseFinished, turn=state.turn, phase="plan",
                           duration_ms=(time.perf_counter() - t_phase) * 1000,
                           summary=PhaseSummary(n_calls=len(calls)))
        return result.model_copy(update={"calls": calls})
```

- [ ] **Step 4: Replace `src/agentic_search/roles/executor.py`**

```python
"""Validates tool calls against manifests, embeds queries, runs ops in parallel, pools hits."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from agentic_search.backends.base import Backend
from agentic_search.backends.filters import filter_fields
from agentic_search.core.emitter import EventEmitter, NullEmitter
from agentic_search.core.hooks import Hooks, SourcePolicy
from agentic_search.core.secrets import scrub
from agentic_search.core.state import CandidatePool, Trace
from agentic_search.core.types import (
    Aggregate,
    FieldType,
    Hit,
    Hybrid,
    Lexical,
    Manifest,
    Query,
    QueryOp,
    Regex,
    TextPart,
    ToolError,
    Vector,
    modality_of,
    required_capabilities,
)
from agentic_search.embedders.base import EmbedderRegistry
from agentic_search.events import ToolCallFinished, ToolCallStarted, ToolErrorInfo
from agentic_search.models.base import ToolCall
from agentic_search.roles.tools import DiscoverRequest, parse_call

_COLLECTION_SCOPED = {"lexical", "vector", "hybrid", "filter", "regex", "aggregate"}
MAX_DISCOVER_CHARS = 8000


@dataclass
class ExecResult:
    new_keys: list[str] = field(default_factory=list)
    errors: list[ToolError] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    hits_per_call: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class _Outcome:
    hits: list[Hit] = field(default_factory=list)
    error: ToolError | None = None
    text: str | None = None


def _short(exc: BaseException) -> str:
    return scrub(str(exc))[:500]


class Executor:
    def __init__(self, backends: dict[str, Backend], manifests: dict[str, Manifest],
                 embedders: EmbedderRegistry, *, hooks: Hooks | None = None,
                 policy: SourcePolicy | None = None, call_timeout: float = 30.0,
                 per_source_concurrency: int = 4, max_limit: int = 100, output_hits: int = 10,
                 snippet_chars: int = 240):
        self.backends = backends
        self.manifests = manifests
        self.embedders = embedders
        self.hooks = hooks or Hooks()
        self.policy = policy or SourcePolicy()
        self.call_timeout = call_timeout
        self.max_limit = max_limit
        self.output_hits = output_hits
        self.snippet_chars = snippet_chars
        self._sems = {name: asyncio.Semaphore(per_source_concurrency) for name in backends}

    # ---- validation ---------------------------------------------------------

    def validate(self, op: QueryOp) -> tuple[str, str] | None:
        manifest = self.manifests.get(op.source)
        if manifest is None or op.source not in self.backends:
            return "validation", f"unknown source {op.source!r}; available: {', '.join(sorted(self.manifests))}"
        missing = required_capabilities(op) - manifest.capabilities
        if missing:
            return "validation", (f"source {op.source!r} does not support "
                                  f"{', '.join(sorted(c.value for c in missing))}")
        coll = manifest.resolve_collection(op.collection)
        if coll is None and (op.collection is not None or op.type in _COLLECTION_SCOPED):
            names = ", ".join(c.name for c in manifest.collections)
            if op.collection is not None:
                return "validation", f"unknown collection {op.collection!r} in {op.source!r} (one of: {names})"
            return "validation", f"source {op.source!r} has several collections; pass collection (one of: {names})"
        if coll is None or not coll.fields:
            return None
        known = {f.name for f in coll.fields}
        referenced = filter_fields(op.filter) if op.filter is not None else set()
        if isinstance(op, (Lexical, Regex)) and op.fields:
            referenced |= set(op.fields)
        if isinstance(op, Aggregate):
            referenced |= set(op.group_by)
        if isinstance(op, (Vector, Hybrid)):
            referenced.add(op.field)
        unknown = referenced - known
        if unknown:
            return "validation", (f"unknown field(s) {sorted(unknown)} in collection {coll.name!r}; "
                                  f"known fields: {sorted(known)}")
        if isinstance(op, (Vector, Hybrid)):
            spec = coll.field(op.field)
            if spec is None or spec.type is not FieldType.VECTOR or not spec.embedder_id:
                return "validation", f"field {op.field!r} is not a vector field with a known embedder"
            embedder = self.embedders.get(spec.embedder_id)
            if embedder is None:
                return "embedder", (f"no embedder registered for {spec.embedder_id!r} (vector field "
                                    f"{op.field!r}); refusing to embed the query with a different model")
            part = op.query_content() if isinstance(op, Vector) else TextPart(text=op.text)
            if modality_of(part) not in embedder.modalities:
                return "embedder", f"embedder {embedder.id!r} cannot embed {modality_of(part).value} queries"
        return None

    # ---- execution ----------------------------------------------------------

    async def run(self, calls: list[ToolCall], *, question: Query, turn: int, pool: CandidatePool,
                  trace: Trace, model_id: str | None = None,
                  emitter: EventEmitter | None = None) -> ExecResult:
        em = emitter or NullEmitter()
        outcomes = await asyncio.gather(*(self._run_emitting(c, question, turn, trace, em)
                                          for c in calls))
        result = ExecResult()
        for c, out in zip(calls, outcomes):
            if out.error is not None:
                result.errors.append(out.error)
                result.outputs[c.id] = out.error.render()
                trace.add("tool_error", turn, call_id=c.id, kind=out.error.kind,
                          message=out.error.message)
                continue
            if out.text is not None:
                result.outputs[c.id] = out.text
                continue
            new = pool.add(out.hits, turn=turn, call_id=c.id)
            result.new_keys.extend(new)
            result.hits_per_call[c.id] = [h.key for h in out.hits]
            result.outputs[c.id] = self._render_hits(out.hits, set(new), model_id)
            trace.add("tool_result", turn, call_id=c.id, n_hits=len(out.hits), n_new=len(new))
        return result

    async def _run_emitting(self, c: ToolCall, question: Query, turn: int, trace: Trace,
                            emitter: EventEmitter) -> _Outcome:
        source = c.arguments.get("source")
        emitter.emit(ToolCallStarted, turn=turn, call_id=c.id, name=c.name, arguments=c.arguments,
                     source=source if isinstance(source, str) else None)
        t0 = time.perf_counter()
        out = await self._run_one(c, question, turn, trace)
        err = out.error
        emitter.emit(ToolCallFinished, turn=turn, call_id=c.id, n_hits=len(out.hits),
                     duration_ms=(time.perf_counter() - t0) * 1000,
                     error=None if err is None else ToolErrorInfo(
                         kind=err.kind, message=scrub(err.message), source=err.source))
        return out

    async def _run_one(self, c: ToolCall, question: Query, turn: int, trace: Trace) -> _Outcome:
        trace.add("tool_call", turn, call_id=c.id, name=c.name, arguments=c.arguments)
        try:
            parsed = parse_call(c, question)
        except ValueError as exc:
            return _Outcome(error=ToolError(call_id=c.id, kind="validation", message=_short(exc)))
        if isinstance(parsed, DiscoverRequest):
            return self._discover(c.id, parsed)
        problem = self.validate(parsed)
        if problem is not None:
            kind, message = problem
            return _Outcome(error=ToolError(call_id=c.id, source=parsed.source, kind=kind,  # type: ignore[arg-type]
                                            message=message))
        op = parsed.model_copy(update={"limit": min(parsed.limit, self.max_limit)})
        if isinstance(op, (Vector, Hybrid)):
            try:
                op = await asyncio.wait_for(self._embed(op), self.call_timeout)
            except TimeoutError:
                return _Outcome(error=ToolError(call_id=c.id, source=op.source, kind="timeout",
                                                message=f"embedding timed out after {self.call_timeout}s"))
            except Exception as exc:
                return _Outcome(error=ToolError(call_id=c.id, source=op.source, kind="embedder",
                                                message=f"{type(exc).__name__}: {_short(exc)}"))
        try:
            async with self._sems[op.source]:
                hits = await asyncio.wait_for(self.backends[op.source].execute(op), self.call_timeout)
        except TimeoutError:
            return _Outcome(error=ToolError(call_id=c.id, source=op.source, kind="timeout",
                                            message=f"timed out after {self.call_timeout}s"))
        except Exception as exc:
            return _Outcome(error=ToolError(call_id=c.id, source=op.source, kind="backend",
                                            message=f"{type(exc).__name__}: {_short(exc)}"))
        return _Outcome(hits=hits)

    async def _embed(self, op: Vector | Hybrid) -> Vector | Hybrid:
        coll = self.manifests[op.source].resolve_collection(op.collection)
        spec = coll.field(op.field) if coll else None
        embedder = self.embedders.get(spec.embedder_id) if spec and spec.embedder_id else None
        assert embedder is not None  # guaranteed by validate()
        part = op.query_content() if isinstance(op, Vector) else TextPart(text=op.text)
        part = await self.hooks.before_model_call(embedder.id, part)
        [vector] = await embedder.embed([part], "query")
        return op.model_copy(update={"vector": list(vector)})

    def _discover(self, call_id: str, req: DiscoverRequest) -> _Outcome:
        manifest = self.manifests.get(req.source)
        if manifest is None:
            return _Outcome(error=ToolError(call_id=call_id, kind="validation",
                                            message=f"unknown source {req.source!r}"))
        if req.collection is None:
            return _Outcome(text=manifest.summary())
        coll = manifest.resolve_collection(req.collection)
        if coll is None:
            return _Outcome(error=ToolError(call_id=call_id, source=req.source, kind="validation",
                                            message=f"unknown collection {req.collection!r}"))
        return _Outcome(text=coll.model_dump_json(indent=1, exclude_none=True)[:MAX_DISCOVER_CHARS])

    def _render_hits(self, hits: list[Hit], new: set[str], model_id: str | None) -> str:
        lines = [f"{len(hits)} hits ({len(new)} new)"]
        for h in hits[: self.output_hits]:
            if model_id is None or self.policy.allows(h.source, model_id):
                body = h.snippet(self.snippet_chars)
            else:
                body = "(content withheld by source policy)"
            score = f" score={h.raw_score:.3f}" if h.raw_score is not None else ""
            lines.append(f"- {h.key}{score}: {body}")
        if len(hits) > self.output_hits:
            lines.append(f"... {len(hits) - self.output_hits} more")
        return "\n".join(lines)
```

- [ ] **Step 5: Replace `src/agentic_search/roles/analyzer.py`**

```python
"""Judges new candidates, blends scores, and writes the digest the planner reads next turn."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass

from agentic_search.core.hooks import Hooks, SourcePolicy
from agentic_search.core.secrets import scrub
from agentic_search.core.state import Candidate, SearchState
from agentic_search.core.types import ModelUsage
from agentic_search.events import PhaseFinished, PhaseStarted, PhaseSummary
from agentic_search.models.base import Decider, JudgeRequest, ToolCall
from agentic_search.roles.executor import ExecResult


@dataclass
class AnalysisResult:
    n_new: int
    n_new_relevant: int | None
    digest: str
    usage: ModelUsage


class Analyzer:
    def __init__(self, decider: Decider | None, *, hooks: Hooks | None = None,
                 policy: SourcePolicy | None = None, relevant_threshold: float = 0.5,
                 judge_weight: float = 0.9, batch_size: int = 16, timeout: float = 60.0,
                 digest_hits: int = 8, snippet_chars: int = 240):
        self.decider = decider
        self.hooks = hooks or Hooks()
        self.policy = policy or SourcePolicy()
        self.relevant_threshold = relevant_threshold
        self.judge_weight = judge_weight
        self.batch_size = batch_size
        self.timeout = timeout
        self.digest_hits = digest_hits
        self.snippet_chars = snippet_chars
        self._can_judge = decider is not None

    def is_relevant(self, cand: Candidate) -> bool:
        return cand.judged and cand.p_relevant is not None and cand.p_relevant >= self.relevant_threshold

    async def judge_keys(self, state: SearchState, keys: list[str]) -> ModelUsage:
        """Judge the unjudged `keys`, bracketed by a `judge` phase on the state's emitter."""
        state.emitter.emit(PhaseStarted, turn=state.turn, phase="judge")
        t0 = time.perf_counter()
        usage, n_judged, n_relevant, error = await self._judge(state, keys)
        state.emitter.emit(PhaseFinished, turn=state.turn, phase="judge",
                           duration_ms=(time.perf_counter() - t0) * 1000,
                           summary=PhaseSummary(n_judged=n_judged, n_relevant=n_relevant,
                                                error=error))
        return usage

    async def _judge(self, state: SearchState,
                     keys: list[str]) -> tuple[ModelUsage, int, int, str | None]:
        """Returns (usage, n_judged, n_relevant, last batch error)."""
        usage = ModelUsage()
        n_judged = n_relevant_total = 0
        error: str | None = None
        pending = [k for k in keys if not state.pool[k].judged]
        if not pending:
            return usage, 0, 0, None
        if not self._can_judge or self.decider is None:
            for k in pending:
                state.pool[k].unjudged_reason = "no judge configured"
            return usage, 0, 0, None
        decider = self.decider
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start:start + self.batch_size]
            try:
                hits = self.policy.redact([state.pool[k].hit for k in batch], decider.id)
                request = await self.hooks.before_model_call(
                    decider.id, JudgeRequest(question=state.question, hits=hits))
                t0 = time.perf_counter()
                result = await asyncio.wait_for(decider.judge(request.question, request.hits),
                                                self.timeout)
            except NotImplementedError:
                self._can_judge = False
                for k in pending[start:]:
                    state.pool[k].unjudged_reason = f"{decider.id} does not judge"
                return usage, n_judged, n_relevant_total, error
            except Exception as exc:
                for k in batch:
                    state.pool[k].unjudged_reason = f"judge failed: {type(exc).__name__}"
                error = scrub(f"{type(exc).__name__}: {exc}")[:300]
                state.trace.add("judge_error", state.turn, decider=decider.id, n=len(batch),
                                error=error)
                continue
            usage = usage.plus(result.usage)
            by_key = {j.key: j for j in result.judgments}
            n_relevant = 0
            for k in batch:
                cand, j = state.pool[k], by_key.get(k)
                if j is None:
                    cand.unjudged_reason = "no judgment returned"
                    continue
                cand.p_relevant, cand.rationale = j.p_relevant, j.rationale
                cand.judged, cand.unjudged_reason = True, None
                n_judged += 1
                n_relevant += self.is_relevant(cand)
            n_relevant_total += n_relevant
            state.trace.add("judge", state.turn, duration_ms=(time.perf_counter() - t0) * 1000,
                            decider=decider.id, n=len(batch), n_relevant=n_relevant)
        return usage, n_judged, n_relevant_total, error

    async def analyze(self, state: SearchState, calls: list[ToolCall], result: ExecResult, *,
                      digest_model_id: str) -> AnalysisResult:
        usage = await self.judge_keys(state, result.new_keys)
        judged_new = [state.pool[k] for k in result.new_keys if state.pool[k].judged]
        n_rel = sum(self.is_relevant(c) for c in judged_new) if judged_new else None
        digest = self.render_digest(state, calls, result, digest_model_id)
        return AnalysisResult(n_new=len(result.new_keys), n_new_relevant=n_rel, digest=digest,
                              usage=usage)

    def render_digest(self, state: SearchState, calls: list[ToolCall], result: ExecResult,
                      model_id: str) -> str:
        """The turn digest as `model_id` may see it (source policy applied)."""
        pool = state.pool
        new = [pool[k] for k in result.new_keys]
        relevant = [c for c in new if self.is_relevant(c)]
        irrelevant = [c for c in new if c.judged and not self.is_relevant(c)]
        header = f"Turn {state.turn}: {len(calls)} calls, {len(new)} new candidates"
        if any(c.judged for c in new):
            header += f", {len(relevant)} judged relevant"
        lines = [f"{header}. Pool size: {len(pool)}.", "Per call:"]
        errors = {e.call_id: e for e in result.errors}
        new_set = set(result.new_keys)
        for c in calls:
            args = json.dumps(c.arguments, default=str)[:200]
            if c.id in errors:
                lines.append(f"- {c.name} {args} -> {errors[c.id].render()}")
                continue
            keys = result.hits_per_call.get(c.id)
            if keys is None:
                lines.append(f"- {c.name} {args} -> ok")
                continue
            line = f"- {c.name} {args} -> {len(keys)} hits, {len(new_set.intersection(keys))} new"
            if self._can_judge:
                line += f", {sum(self.is_relevant(pool[k]) for k in keys)} relevant"
            lines.append(line)
        if relevant:
            lines.append("New relevant:")
            relevant.sort(key=lambda c: -(c.p_relevant or 0.0))
            lines.extend(self._line(c, model_id) for c in relevant[: self.digest_hits])
        if irrelevant:
            lines.append("Judged not relevant (examples):")
            lines.extend(self._line(c, model_id) for c in irrelevant[:3])
        if new and not relevant and not irrelevant:
            lines.append("New candidates (not judged):")
            lines.extend(self._line(c, model_id) for c in new[: self.digest_hits])
        top = pool.ranked(self.judge_weight)[:5]
        if top:
            lines.append("Current top results: " + ", ".join(f"{c.hit.key} ({s:.2f})" for c, s in top))
        return "\n".join(lines)

    def _line(self, cand: Candidate, model_id: str) -> str:
        h = cand.hit
        allowed = self.policy.allows(h.source, model_id)
        body = h.snippet(self.snippet_chars) if allowed else "(content withheld by source policy)"
        p = f" p={cand.p_relevant:.2f}" if cand.judged and cand.p_relevant is not None else ""
        why = f" ({cand.rationale})" if cand.rationale and allowed else ""
        return f"- {h.key}{p}: {body}{why}"
```

- [ ] **Step 6: Replace `src/agentic_search/roles/controller.py`**

```python
"""Decides whether to keep searching. Hard budgets always win over any decider."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from agentic_search.core.hooks import Hooks
from agentic_search.core.secrets import scrub
from agentic_search.core.state import SearchState
from agentic_search.core.types import StopReason
from agentic_search.events import PhaseFinished, PhaseStarted, PhaseSummary
from agentic_search.models.base import Action, ControllerView, Decider, Decision


class Controller:
    def __init__(self, decider: Decider | None, *, hooks: Hooks | None = None,
                 relevant_threshold: float = 0.5, min_new_relevant: int = 1,
                 stable_top_k: int = 10, timeout: float = 60.0):
        self.decider = decider
        self.hooks = hooks or Hooks()
        self.relevant_threshold = relevant_threshold
        self.min_new_relevant = min_new_relevant
        self.stable_top_k = stable_top_k
        self.timeout = timeout
        self._can_decide = decider is not None

    def budget_stop(self, state: SearchState) -> StopReason | None:
        b, u = state.budget, state.usage
        if u.turns >= b.max_turns:
            return StopReason.BUDGET_TURNS
        if u.tool_calls >= b.max_tool_calls:
            return StopReason.BUDGET_TOOL_CALLS
        if b.max_tokens is not None and u.total_tokens >= b.max_tokens:
            return StopReason.BUDGET_TOKENS
        if b.max_cost_usd is not None and u.cost_usd >= b.max_cost_usd:
            return StopReason.BUDGET_COST
        if b.max_seconds is not None and state.elapsed() >= b.max_seconds:
            return StopReason.BUDGET_TIME
        return None

    def budget_remaining(self, state: SearchState) -> dict[str, Any]:
        b, u = state.budget, state.usage
        return {
            "turns": b.max_turns - u.turns,
            "tool_calls": b.max_tool_calls - u.tool_calls,
            "tokens": None if b.max_tokens is None else b.max_tokens - u.total_tokens,
            "cost_usd": None if b.max_cost_usd is None else round(b.max_cost_usd - u.cost_usd, 4),
            "seconds": None if b.max_seconds is None else round(b.max_seconds - state.elapsed(), 1),
        }

    def total_relevant(self, state: SearchState) -> int | None:
        judged = [c for c in state.pool.candidates() if c.judged and c.p_relevant is not None]
        if not judged:
            return None
        return sum(c.p_relevant >= self.relevant_threshold for c in judged)  # type: ignore[operator]

    def heuristic(self, state: SearchState) -> Decision:
        if not state.history:
            return Decision(action=Action.CONTINUE, note="no turns yet")
        last = state.history[-1]
        if last.n_calls and last.n_errors == last.n_calls:
            return Decision(action=Action.REFINE, note="every call failed; fix the errors")
        if last.n_new_relevant is not None:
            if last.n_new_relevant < self.min_new_relevant:
                if (self.total_relevant(state) or 0) > 0:
                    return Decision(action=Action.STOP, note="no new relevant results this turn")
                return Decision(action=Action.BROADEN, note="nothing relevant yet; broaden or rephrase")
        elif last.n_new == 0:
            return Decision(action=Action.STOP, note="no new candidates")
        if len(state.history) >= 2:
            k = self.stable_top_k
            prev, cur = state.history[-2].top_keys[:k], last.top_keys[:k]
            if cur and cur == prev:
                return Decision(action=Action.STOP, note="top results stable")
        return Decision(action=Action.CONTINUE, note="new results found; keep exploring")

    async def decide(self, state: SearchState, *, digest: str | None = None) -> Decision:
        """`digest` is the turn digest rendered for this controller's decider (default state.digest)."""
        state.emitter.emit(PhaseStarted, turn=state.turn, phase="decide")
        t0 = time.perf_counter()
        by = "heuristic"
        error: str | None = None
        if self.decider is None or not self._can_decide:
            decision = self.heuristic(state)
        else:
            decider = self.decider
            view = ControllerView(question=state.question, turn=state.turn, history=state.history,
                                  digest=state.digest if digest is None else digest,
                                  total_relevant=self.total_relevant(state),
                                  budget_remaining=self.budget_remaining(state))
            try:
                view = await self.hooks.before_model_call(decider.id, view)
                decision = await asyncio.wait_for(decider.decide(view), self.timeout)
                by = decider.id
            except NotImplementedError:
                self._can_decide = False
                decision = self.heuristic(state)
            except Exception as exc:
                error = scrub(f"{type(exc).__name__}: {exc}")[:300]
                state.trace.add("decision_error", state.turn, decider=decider.id, error=error)
                decision = self.heuristic(state)
        state.trace.add("decision", state.turn, action=decision.action.value, note=decision.note,
                        confidence=decision.confidence, by=by)
        state.emitter.emit(PhaseFinished, turn=state.turn, phase="decide",
                           duration_ms=(time.perf_counter() - t0) * 1000,
                           summary=PhaseSummary(action=decision.action.value,
                                                confidence=decision.confidence, error=error))
        return decision
```

- [ ] **Step 7: Run the tests**

Run: `uv run pytest -q tests/roles/test_role_events.py`
Expected: `6 passed`.

Run: `uv run pytest -q && uv run ruff check`
Expected: `363 passed, 5 skipped, 79 deselected` and `All checks passed!`.

- [ ] **Step 8: Commit**

```bash
git add src/agentic_search/roles tests/roles/test_role_events.py
git commit -m "feat(roles): emit plan, tool-call, judge and decide events

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `Harness.stream()` and `search()` on top of it

**Files:**
- Modify: `src/agentic_search/core/harness.py`
- Test: `tests/test_stream.py`

**Interfaces:**
- Consumes (Tasks 1–2): the event classes, `EventEmitter`, `TERMINAL_EVENTS`, `Executor.run(..., emitter=)`, and the role phase events.
- Produces:
  - `Harness.stream(question, *, sources=None, top_k=20, mode=None, budget=None, snapshot_k=10, include_content=False) -> SearchStream`. It raises `HarnessError` before any event for an unknown mode, `top_k < 1` or `snapshot_k < 0`.
  - `SearchStream`: an async iterator of `SearchEvent` and an async context manager. `aclose()`, leaving the `async with` block, or cancelling the consuming task cancels the search task.
  - `Harness.search(...)`: same signature as before, now rejecting `top_k < 1` too. It drains `_events(...)` with `snapshot_k=0`, returns `SearchFinished.result`, and otherwise re-raises the original exception.
  - Harness-level events:
    - `search_started`, emitted after setup and source selection;
    - the `query` phase around `executor.run`;
    - the `delegate` phase;
    - `results_updated` + `usage_updated` after each turn's analysis, after each delegate tool round, and once after the delegate finishes;
    - `usage_updated` after each decide;
    - `search_finished` / `search_failed`.
  - `_DelegateRuntime(..., on_round: Callable[[], None])`, called after each executed round, before `state.turn` increments.

Design notes for the implementer:
- `stream()` is a plain method that validates and returns `SearchStream(self._events(...))`, so validation errors raise at call time, not on first iteration.
- The queue is unbounded on purpose: the number of events is bounded by the budget, and a bounded queue would let a slow consumer stall backend calls.
- Snapshots always rank by pool score (`state.pool.ranked(judge_weight)`), including in model mode, where the final result can then reorder by the delegate's ranking.

- [ ] **Step 1: Write the failing test**

Create `tests/test_stream.py`:

```python
import asyncio

import pytest

from agentic_search import Budget, Harness, HarnessError, HarnessSettings
from agentic_search.core.secrets import register_secret
from agentic_search.core.types import StopReason
from agentic_search.events import (
    PhaseFinished,
    PhaseStarted,
    ResultsUpdated,
    SearchFailed,
    SearchFinished,
    SearchStarted,
    ToolCallFinished,
    ToolCallStarted,
    UsageUpdated,
)
from agentic_search.models.base import Action
from agentic_search.testing import (
    FailingDecider,
    KeywordJudge,
    ScriptedController,
    ScriptedDriver,
    call,
)


def lex(text, **kw):
    return call("lexical_search", source="docs", text=text, **kw)


def make(docs_backend, driver, **kw):
    return Harness([docs_backend], driver, embedders=[docs_backend.embedder], **kw)


async def collect(stream):
    return [ev async for ev in stream]


def shape(events):
    """Compact sequence: event type, plus phase for phase events."""
    out = []
    for e in events:
        out.append(f"{e.type}:{e.phase}" if isinstance(e, (PhaseStarted, PhaseFinished)) else e.type)
    return out


def assert_well_formed(events):
    assert [e.seq for e in events] == list(range(len(events)))
    assert len({e.search_id for e in events}) == 1
    terminal = [e for e in events if isinstance(e, (SearchFinished, SearchFailed))]
    assert terminal == [events[-1]]
    started = {e.call_id for e in events if isinstance(e, ToolCallStarted)}
    finished = {e.call_id for e in events if isinstance(e, ToolCallFinished)}
    assert started == finished
    assert [e.at_ms for e in events] == sorted(e.at_ms for e in events)


async def test_retrieval_mode_sequence(docs_backend):
    h = make(docs_backend, ScriptedDriver([[lex("headache")]]), analyzer=KeywordJudge(["headache"]))
    events = await collect(h.stream("q", mode="retrieval"))
    assert_well_formed(events)
    assert shape(events) == [
        "search_started", "phase_started:plan", "phase_finished:plan", "phase_started:query",
        "tool_call_started", "tool_call_finished", "phase_finished:query",
        "phase_started:judge", "phase_finished:judge", "results_updated", "usage_updated",
        "search_finished"]
    started = events[0]
    assert isinstance(started, SearchStarted) and started.sources == ["docs"]
    assert started.mode == "retrieval"
    query_done = events[6]
    assert query_done.summary.n_calls == 1 and query_done.summary.n_new == 2
    assert events[-1].stop_reason is StopReason.SINGLE_PASS


async def test_harness_mode_sequence_and_final_snapshot(docs_backend):
    driver = ScriptedDriver([[lex("headache")], [lex("fever")]])
    h = make(docs_backend, driver, analyzer=KeywordJudge(["headache", "fever"]),
             controller=ScriptedController([Action.CONTINUE]))
    events = await collect(h.stream("q", snapshot_k=3))
    assert_well_formed(events)
    turn = ["phase_started:plan", "phase_finished:plan", "phase_started:query",
            "tool_call_started", "tool_call_finished", "phase_finished:query",
            "phase_started:judge", "phase_finished:judge", "results_updated", "usage_updated",
            "phase_started:decide", "phase_finished:decide", "usage_updated"]
    assert shape(events) == ["search_started", *turn, *turn, "search_finished"]
    assert [e.turn for e in events if isinstance(e, ResultsUpdated)] == [0, 1]
    decides = [e for e in events if isinstance(e, PhaseFinished) and e.phase == "decide"]
    assert [d.summary.action for d in decides] == ["continue", "stop"]
    last = [e for e in events if isinstance(e, ResultsUpdated)][-1]
    final = events[-1].result
    assert len(last.hits) <= 3
    assert [(s.key, s.score) for s in last.hits] == [(r.hit.key, r.score) for r in final.hits[:3]]
    assert last.pool_size == 2 and last.n_relevant == 2
    usage = [e for e in events if isinstance(e, UsageUpdated)][-1].usage
    assert usage.model_dump() == final.usage.model_dump()


async def test_model_mode_sequence(docs_backend):
    driver = ScriptedDriver(delegate_calls=[[lex("headache")], [lex("fever")]],
                            delegate_keys=["docs:d4", "docs:d1"])
    h = make(docs_backend, driver, analyzer=KeywordJudge(["headache"]))
    events = await collect(h.stream("q", mode="model"))
    assert_well_formed(events)
    rnd = ["tool_call_started", "tool_call_finished", "results_updated", "usage_updated"]
    assert shape(events) == [
        "search_started", "phase_started:delegate", *rnd, *rnd, "phase_finished:delegate",
        "phase_started:judge", "phase_finished:judge", "results_updated", "usage_updated",
        "search_finished"]
    delegate_done = events[10]
    assert delegate_done.summary.n_ranked == 2
    assert set(events[-1].result.keys()) == {"docs:d1", "docs:d4"}


async def test_parallel_calls_pair_and_stay_ordered(docs_backend):
    calls = [lex("headache"), lex("fever"), lex("castles"), call("lexical_search", source="nope", text="x")]
    h = make(docs_backend, ScriptedDriver([calls]))
    events = await collect(h.stream("q", mode="retrieval"))
    assert_well_formed(events)
    finished = [e for e in events if isinstance(e, ToolCallFinished)]
    assert len(finished) == 4
    assert sum(e.error is not None for e in finished) == 1
    assert events[-1].type == "search_finished"


async def test_snapshot_k_zero_and_include_content(docs_backend):
    h = make(docs_backend, ScriptedDriver([[lex("headache")]]))
    events = await collect(h.stream("q", mode="retrieval", snapshot_k=0))
    assert not any(isinstance(e, ResultsUpdated) for e in events)
    h2 = make(docs_backend, ScriptedDriver([[lex("headache")]]))
    plain = [e for e in await collect(h2.stream("q", mode="retrieval")) if isinstance(e, ResultsUpdated)]
    assert plain[0].hits and all(s.content is None for s in plain[0].hits)
    h3 = make(docs_backend, ScriptedDriver([[lex("headache")]]))
    full = [e for e in await collect(h3.stream("q", mode="retrieval", include_content=True))
            if isinstance(e, ResultsUpdated)]
    assert all(s.content and s.content[0].text for s in full[0].hits)


async def test_invalid_arguments_raise_before_any_event(docs_backend):
    h = make(docs_backend, ScriptedDriver([]))
    for kwargs in ({"mode": "psychic"}, {"top_k": 0}, {"snapshot_k": -1}):
        with pytest.raises(HarnessError):
            h.stream("q", **kwargs)
    with pytest.raises(HarnessError):
        await h.search("q", top_k=0)


async def test_unknown_source_fails_stream_and_search_raises(docs_backend):
    h = make(docs_backend, ScriptedDriver([]))
    events = await collect(h.stream("q", sources=["nope"]))
    assert shape(events) == ["search_failed"]
    assert events[0].error_type == "HarnessError" and "nope" in events[0].message
    with pytest.raises(HarnessError, match="unknown or undiscovered sources"):
        await h.search("q", sources=["nope"])


async def test_unexpected_error_fails_stream_and_search_reraises(docs_backend):
    class Boom(Exception):
        pass

    class ExplodingDriver(ScriptedDriver):
        async def plan(self, view, tools):
            raise Boom("planner crashed")

    h = make(docs_backend, ExplodingDriver())
    events = await collect(h.stream("q"))
    assert shape(events) == ["search_started", "phase_started:plan", "search_failed"]
    assert events[-1].error_type == "Boom"
    with pytest.raises(Boom):
        await h.search("q")


async def test_backend_and_decider_failures_do_not_end_stream(docs_backend):
    async def broken(op):
        raise RuntimeError("index offline")

    docs_backend.execute = broken
    h = make(docs_backend, ScriptedDriver([[lex("headache")]]), controller=FailingDecider())
    events = await collect(h.stream("q"))
    assert events[-1].type == "search_finished"
    [done] = [e for e in events if isinstance(e, ToolCallFinished)]
    assert done.error.kind == "backend" and "index offline" in done.error.message


async def test_secrets_never_appear_in_events(docs_backend):
    register_secret("sk-live-7777")

    async def leaky(op):
        raise RuntimeError("auth failed for sk-live-7777")

    docs_backend.execute = leaky
    h = make(docs_backend, ScriptedDriver([[lex("sk-live-7777 headache")]]))
    events = await collect(h.stream("q"))
    assert all("sk-live-7777" not in e.model_dump_json() for e in events)


async def test_secret_in_document_snippet_is_scrubbed(docs_backend):
    register_secret("Aspirin")  # stands in for a secret that appears inside a document
    h = make(docs_backend, ScriptedDriver([[lex("headache")]]))
    events = await collect(h.stream("q", mode="retrieval"))
    [snap] = [e for e in events if isinstance(e, ResultsUpdated)]
    assert all("Aspirin" not in s.snippet for s in snap.hits)


async def test_cancel_by_closing_stream_cancels_backend_call(docs_backend):
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def slow(op):
        entered.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return []

    docs_backend.execute = slow
    # A long call timeout, so only the stream's own cancellation can stop the backend call.
    h = make(docs_backend, ScriptedDriver([[lex("headache")]]),
             settings=HarnessSettings(call_timeout=600))
    seen = []

    async def consume():
        async with h.stream("q") as stream:
            async for ev in stream:
                seen.append(ev)
                if isinstance(ev, ToolCallStarted):
                    await entered.wait()
                    break

    await asyncio.wait_for(consume(), 5)
    assert cancelled.is_set()
    assert seen[-1].type == "tool_call_started"


async def test_cancel_by_cancelling_consumer_task(docs_backend):
    cancelled = asyncio.Event()

    async def slow(op):
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    docs_backend.execute = slow
    h = make(docs_backend, ScriptedDriver([[lex("headache")]]),
             settings=HarnessSettings(call_timeout=600))

    async def consume():
        async for _ in h.stream("q"):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    assert cancelled.is_set()


async def test_concurrent_streams_are_isolated_and_discover_once(docs_backend):
    calls = 0
    original = docs_backend.discover

    async def counting(*a, **kw):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return await original(*a, **kw)

    docs_backend.discover = counting
    h = make(docs_backend, ScriptedDriver([[lex("headache")], [lex("fever")]]))
    a, b = await asyncio.gather(collect(h.stream("q", mode="retrieval")),
                                collect(h.stream("q", mode="retrieval")))
    assert calls == 1
    assert_well_formed(a)
    assert_well_formed(b)
    assert a[0].search_id != b[0].search_id


async def test_search_matches_stream_result(docs_backend):
    def build():
        return make(docs_backend, ScriptedDriver([[lex("headache")], [lex("fever")]]),
                    analyzer=KeywordJudge(["headache", "fever"]))

    streamed = (await collect(build().stream("q", budget=Budget(max_turns=2))))[-1].result
    direct = await build().search("q", budget=Budget(max_turns=2))
    assert streamed.keys() == direct.keys()
    assert [h.score for h in streamed.hits] == [h.score for h in direct.hits]
    assert streamed.stop_reason is direct.stop_reason
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -q tests/test_stream.py`
Expected: `15 failed`, with `AttributeError: 'Harness' object has no attribute 'stream'` (the `top_k=0` check in `test_invalid_arguments_raise_before_any_event` is reached only after that).

- [ ] **Step 3: Replace `src/agentic_search/core/harness.py`**

```python
"""The Harness: Plan → Execute → Analyze → Decide over any backends, with swappable models."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
import uuid
from typing import Any, AsyncIterator, Callable, Literal, Sequence

from pydantic import BaseModel

from agentic_search.backends.base import Backend
from agentic_search.core.annotations import apply_annotations
from agentic_search.core.emitter import EventEmitter
from agentic_search.core.hooks import Hooks, SourcePolicy
from agentic_search.core.result import RankedHit, SearchResult
from agentic_search.core.secrets import scrub
from agentic_search.core.state import Candidate, SearchState
from agentic_search.core.types import Budget, Manifest, ModelUsage, Query, StopReason
from agentic_search.embedders.base import Embedder, EmbedderRegistry
from agentic_search.events import (
    TERMINAL_EVENTS,
    HitSummary,
    PhaseFinished,
    PhaseStarted,
    PhaseSummary,
    ResultsUpdated,
    SearchEvent,
    SearchFailed,
    SearchFinished,
    SearchStarted,
    UsageUpdated,
)
from agentic_search.models.base import (
    Action,
    Decider,
    DelegateRequest,
    Driver,
    ToolCall,
    ToolSpec,
    TurnSummary,
)
from agentic_search.roles.analyzer import Analyzer
from agentic_search.roles.controller import Controller
from agentic_search.roles.executor import Executor
from agentic_search.roles.planner import Planner, render_manifests
from agentic_search.roles.tools import build_tool_specs

Mode = Literal["retrieval", "harness", "model"]
_MODES = ("retrieval", "harness", "model")
__all__ = ["Harness", "HarnessError", "HarnessSettings", "Mode", "RankedHit", "SearchResult",
           "SearchStream"]


class HarnessError(Exception):
    """Harness-level failure: misconfiguration or no reachable backends."""


class HarnessSettings(BaseModel):
    max_calls_per_turn: int = 8
    call_timeout: float = 30.0
    per_source_concurrency: int = 4
    max_limit: int = 100
    relevant_threshold: float = 0.5
    judge_weight: float = 0.9
    judge_batch_size: int = 16
    decider_timeout: float = 60.0
    min_new_relevant: int = 1
    discover_timeout: float | None = 300.0


class _Options(BaseModel):
    question: Query
    sources: list[str] | None
    top_k: int
    mode: str
    budget: Budget
    snapshot_k: int
    include_content: bool


class SearchStream:
    """Async iterator over one search's events; also an async context manager.

    Leaving the `async with` block, calling `aclose()`, or cancelling the consuming task cancels
    the search. Dropping the stream mid-iteration also cancels it once the generator is
    garbage-collected; use `async with` when cancellation must be immediate."""

    def __init__(self, gen: AsyncIterator[SearchEvent]):
        self._gen = gen

    def __aiter__(self) -> SearchStream:
        return self

    async def __anext__(self) -> SearchEvent:
        return await self._gen.__anext__()

    async def aclose(self) -> None:
        await self._gen.aclose()  # type: ignore[attr-defined]

    async def __aenter__(self) -> SearchStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


class _DelegateRuntime:
    """ToolRuntime for model-centric drivers: executes calls, enforces budgets, records the trace."""

    def __init__(self, state: SearchState, executor: Executor, controller: Controller, model_id: str,
                 hooks: Hooks, on_round: Callable[[], None] = lambda: None):
        self.state, self.executor, self.controller, self.model_id = state, executor, controller, model_id
        self.hooks = hooks
        self.on_round = on_round
        self.exhausted: StopReason | None = None
        self.reported = ModelUsage()

    def report_usage(self, usage: ModelUsage) -> None:
        self.state.usage.add_model(usage)
        self.reported = self.reported.plus(usage)

    def budget_exhausted(self) -> bool:
        reason = self.exhausted or self.controller.budget_stop(self.state)
        if reason is not None and self.exhausted is None:
            self.exhausted = reason
        return reason is not None

    def unreported(self, total: ModelUsage) -> ModelUsage:
        """The part of a driver's final usage total not already reported mid-run."""
        r = self.reported
        return ModelUsage(input_tokens=max(0, total.input_tokens - r.input_tokens),
                          output_tokens=max(0, total.output_tokens - r.output_tokens),
                          cost_usd=max(0.0, total.cost_usd - r.cost_usd))

    async def call(self, calls: list[ToolCall]) -> list[str]:
        state = self.state
        reason = self.controller.budget_stop(state)
        allowed: list[ToolCall] = []
        if reason is None:
            allowed = calls[: max(0, state.budget.max_tool_calls - state.usage.tool_calls)]
            if len(allowed) < len(calls):
                reason = StopReason.BUDGET_TOOL_CALLS
        if reason is not None and self.exhausted is None:
            self.exhausted = reason
        message = (f"[budget] {reason.value}: stop searching and call finish with your ranked keys."
                   if reason is not None else "")
        outputs: dict[str, str] = {}
        if allowed:
            result = await self.executor.run(allowed, question=state.question, turn=state.turn,
                                             pool=state.pool, trace=state.trace,
                                             model_id=self.model_id, emitter=state.emitter)
            state.usage.tool_calls += len(allowed)
            state.usage.turns += 1
            self.on_round()
            state.turn += 1
            outputs = result.outputs
        texts = [outputs.get(c.id, message) for c in calls]
        try:
            return list(await self.hooks.before_model_call(self.model_id, texts))
        except Exception as exc:
            state.trace.add("hook_error", state.turn, model=self.model_id, n=len(calls),
                            error=f"{type(exc).__name__}: {str(exc)[:300]}")
            return [f"[withheld by hook] {type(exc).__name__}" for _ in calls]


class Harness:
    def __init__(self, backends: Sequence[Backend], driver: Driver, *,
                 embedders: Sequence[Embedder] = (), analyzer: Decider | None = None,
                 controller: Decider | None = None, mode: Mode = "harness",
                 budget: Budget | None = None, hooks: Hooks | None = None,
                 source_policy: dict[str, set[str]] | None = None,
                 annotations: dict[str, dict[str, Any]] | None = None,
                 settings: HarnessSettings | None = None):
        if not backends:
            raise HarnessError("at least one backend is required")
        names = [b.name for b in backends]
        if len(set(names)) != len(names):
            raise HarnessError(f"duplicate backend names: {names}")
        if mode not in _MODES:
            raise HarnessError(f"unknown mode {mode!r}; one of {_MODES}")
        self.backends = {b.name: b for b in backends}
        self.annotations = annotations or {}
        unknown = set(self.annotations) - set(self.backends)
        if unknown:
            raise HarnessError(f"annotations for unknown sources: {sorted(unknown)}")
        self.driver = driver
        self.embedders = EmbedderRegistry(embedders)
        self.analyzer_decider = analyzer
        self.controller_decider = controller
        self.mode: Mode = mode
        self.budget = budget or Budget()
        self.hooks = hooks or Hooks()
        self.policy = SourcePolicy(source_policy)
        self.settings = settings or HarnessSettings()
        for backend in backends:  # backends that embed their own content get hooked embedders
            bind = getattr(backend, "bind_hooks", None)
            if callable(bind):
                bind(self.hooks, self.policy)
        self.manifests: dict[str, Manifest] = {}
        self.setup_errors: dict[str, str] = {}
        self._ready = False
        self._setup_lock = asyncio.Lock()

    async def setup(self) -> dict[str, Manifest]:
        async with self._setup_lock:
            if self._ready:
                return self.manifests
            names = list(self.backends)
            timeout = self.settings.discover_timeout
            results = await asyncio.gather(
                *(asyncio.wait_for(self.backends[n].discover("full"), timeout) for n in names),
                return_exceptions=True)
            for name, res in zip(names, results):
                if isinstance(res, BaseException):
                    self.setup_errors[name] = scrub(f"{type(res).__name__}: {res}")
                    continue
                self.manifests[name] = apply_annotations(res, self.annotations.get(name))
            if not self.manifests:
                raise HarnessError(f"no backend could be discovered: {self.setup_errors}")
            self._ready = True
            return self.manifests

    def stream(self, question: str | Query, *, sources: list[str] | None = None,
               top_k: int = 20, mode: Mode | None = None, budget: Budget | None = None,
               snapshot_k: int = 10, include_content: bool = False) -> SearchStream:
        """Run one search, yielding typed events as it progresses. Invalid arguments raise
        HarnessError here, before any event; every other failure ends the stream with a
        `search_failed` event. The last event is `search_finished` or `search_failed`."""
        opts = self._options(question, sources, top_k, mode, budget, snapshot_k, include_content)
        return SearchStream(self._events(opts, []))

    async def search(self, question: str | Query, *, sources: list[str] | None = None,
                     top_k: int = 20, mode: Mode | None = None,
                     budget: Budget | None = None) -> SearchResult:
        opts = self._options(question, sources, top_k, mode, budget, 0, False)
        failure: list[BaseException] = []
        async with contextlib.aclosing(self._events(opts, failure)) as events:
            async for event in events:
                if isinstance(event, SearchFinished):
                    return event.result
        raise failure[0]

    def _options(self, question: str | Query, sources: list[str] | None, top_k: int,
                 mode: Mode | None, budget: Budget | None, snapshot_k: int,
                 include_content: bool) -> _Options:
        run_mode = mode or self.mode
        if run_mode not in _MODES:
            raise HarnessError(f"unknown mode {run_mode!r}; one of {_MODES}")
        if top_k < 1:
            raise HarnessError(f"top_k must be at least 1, got {top_k}")
        if snapshot_k < 0:
            raise HarnessError(f"snapshot_k must be 0 or more, got {snapshot_k}")
        return _Options(question=Query.of(question) if isinstance(question, str) else question,
                        sources=sources, top_k=top_k, mode=run_mode, budget=budget or self.budget,
                        snapshot_k=snapshot_k, include_content=include_content)

    async def _events(self, opts: _Options,
                      failure: list[BaseException]) -> AsyncIterator[SearchEvent]:
        queue: asyncio.Queue[SearchEvent] = asyncio.Queue()
        emitter = EventEmitter(uuid.uuid4().hex, queue.put_nowait)
        task = asyncio.create_task(self._run_search(opts, emitter, failure))
        try:
            while True:
                event = await queue.get()
                yield event
                if isinstance(event, TERMINAL_EVENTS):
                    break
        finally:
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _run_search(self, opts: _Options, emitter: EventEmitter,
                          failure: list[BaseException]) -> None:
        turn = 0
        try:
            await self.setup()
            manifests = self._select(opts.sources)
            state = SearchState(question=opts.question, manifests=manifests, budget=opts.budget,
                                emitter=emitter)
            state.trace.set_listener(self.hooks.on_trace_event)
            state.trace.add("setup", 0, sources=sorted(manifests), setup_errors=self.setup_errors,
                            mode=opts.mode)
            emitter.emit(SearchStarted, turn=0, question=opts.question, mode=opts.mode,
                         sources=sorted(manifests), budget=opts.budget,
                         setup_errors=dict(self.setup_errors))
            result = await self._run_state(state, opts)
            turn = state.turn
            emitter.emit(SearchFinished, turn=turn, result=result, stop_reason=result.stop_reason)
        except Exception as exc:
            failure.append(exc)
            emitter.emit(SearchFailed, turn=turn, error_type=type(exc).__name__,
                         message=scrub(str(exc))[:500])

    async def _run_state(self, state: SearchState, opts: _Options) -> SearchResult:
        manifests = state.manifests
        s = self.settings
        executor = Executor({n: self.backends[n] for n in manifests}, manifests, self.embedders,
                            hooks=self.hooks, policy=self.policy, call_timeout=s.call_timeout,
                            per_source_concurrency=s.per_source_concurrency, max_limit=s.max_limit)
        analyzer = Analyzer(self.analyzer_decider, hooks=self.hooks, policy=self.policy,
                            relevant_threshold=s.relevant_threshold, judge_weight=s.judge_weight,
                            batch_size=s.judge_batch_size, timeout=s.decider_timeout)
        controller = Controller(self.controller_decider, hooks=self.hooks,
                                relevant_threshold=s.relevant_threshold,
                                min_new_relevant=s.min_new_relevant, timeout=s.decider_timeout)
        tools = build_tool_specs(manifests)

        def progress() -> None:
            if opts.snapshot_k > 0:
                self._emit_snapshot(state, analyzer, opts)
            state.emitter.emit(UsageUpdated, turn=state.turn, usage=state.usage.model_copy())

        order: list[str] | None = None
        if opts.mode == "model":
            reason, order = await self._run_delegate(state, executor, analyzer, controller, tools,
                                                     progress)
        else:
            reason = await self._run_loop(state, executor, analyzer, controller, tools, progress,
                                          single_pass=opts.mode == "retrieval")
        return self._finalize(state, reason, opts.top_k, order, opts.mode)

    def _emit_snapshot(self, state: SearchState, analyzer: Analyzer, opts: _Options) -> None:
        ranked = state.pool.ranked(self.settings.judge_weight)
        hits = [self._summary(c, score, opts.include_content)
                for c, score in ranked[:opts.snapshot_k]]
        state.emitter.emit(ResultsUpdated, turn=state.turn, hits=hits, pool_size=len(state.pool),
                           n_relevant=sum(analyzer.is_relevant(c) for c, _ in ranked))

    @staticmethod
    def _summary(cand: Candidate, score: float, include_content: bool) -> HitSummary:
        h = cand.hit
        title = h.metadata.get("title")
        return HitSummary(key=h.key, source=h.source, doc_id=h.doc_id,
                          title=scrub(title) if isinstance(title, str) else None,
                          snippet=scrub(h.snippet(300)), score=round(score, 6),
                          p_relevant=cand.p_relevant, judged=cand.judged,
                          first_turn=cand.first_turn,
                          content=list(h.content) if include_content else None)

    def _select(self, sources: list[str] | None) -> dict[str, Manifest]:
        if sources is None:
            return dict(self.manifests)
        unknown = [s for s in sources if s not in self.manifests]
        if unknown:
            raise HarnessError(f"unknown or undiscovered sources: {unknown}")
        return {s: self.manifests[s] for s in sources}

    async def _run_loop(self, state: SearchState, executor: Executor, analyzer: Analyzer,
                        controller: Controller, tools: list[ToolSpec],
                        progress: Callable[[], None], *, single_pass: bool) -> StopReason:
        planner = Planner(self.driver, hooks=self.hooks,
                          max_calls_per_turn=self.settings.max_calls_per_turn)
        while True:
            reason = controller.budget_stop(state)
            if reason is not None:
                state.trace.add("budget", state.turn, reason=reason.value)
                return reason
            plan = await planner.plan(state, tools)
            state.usage.add_model(plan.usage)
            calls = plan.calls[: state.budget.max_tool_calls - state.usage.tool_calls]
            if not calls:
                return StopReason.NO_PLAN
            state.emitter.emit(PhaseStarted, turn=state.turn, phase="query")
            t0 = time.perf_counter()
            result = await executor.run(calls, question=state.question, turn=state.turn,
                                        pool=state.pool, trace=state.trace, model_id=self.driver.id,
                                        emitter=state.emitter)
            state.emitter.emit(PhaseFinished, turn=state.turn, phase="query",
                               duration_ms=(time.perf_counter() - t0) * 1000,
                               summary=PhaseSummary(
                                   n_calls=len(calls), n_new=len(result.new_keys),
                                   n_hits=sum(len(k) for k in result.hits_per_call.values()),
                                   n_errors=len(result.errors)))
            state.usage.tool_calls += len(calls)
            state.usage.turns += 1
            analysis = await analyzer.analyze(state, calls, result, digest_model_id=self.driver.id)
            state.usage.add_model(analysis.usage)
            progress()
            state.digest, state.last_errors = analysis.digest, result.errors
            top = [c.hit.key for c, _ in state.pool.ranked(self.settings.judge_weight)[:20]]
            state.history.append(TurnSummary(
                turn=state.turn, n_calls=len(calls), n_errors=len(result.errors),
                n_new=analysis.n_new, n_new_relevant=analysis.n_new_relevant, top_keys=top))
            if single_pass:
                return StopReason.SINGLE_PASS
            reason = controller.budget_stop(state)
            if reason is not None:
                state.trace.add("budget", state.turn, reason=reason.value)
                return reason
            ctrl_digest = None
            if self.controller_decider is not None:
                ctrl_digest = analyzer.render_digest(state, calls, result, self.controller_decider.id)
            decision = await controller.decide(state, digest=ctrl_digest)
            state.usage.add_model(decision.usage)
            state.emitter.emit(UsageUpdated, turn=state.turn, usage=state.usage.model_copy())
            if decision.action is Action.STOP:
                return StopReason.CONTROLLER_STOP
            state.last_decision = decision
            state.turn += 1

    async def _run_delegate(self, state: SearchState, executor: Executor, analyzer: Analyzer,
                            controller: Controller, tools: list[ToolSpec],
                            progress: Callable[[], None]) -> tuple[StopReason, list[str] | None]:
        runtime = _DelegateRuntime(state, executor, controller, self.driver.id, self.hooks,
                                   on_round=progress)
        state.emitter.emit(PhaseStarted, turn=state.turn, phase="delegate")
        request = DelegateRequest(question=state.question, context=render_manifests(state.manifests))
        request = await self.hooks.before_model_call(self.driver.id, request)
        t0 = time.perf_counter()
        result = await self.driver.run_delegate(request.question, tools, runtime, state.budget,
                                                context=request.context)
        state.usage.add_model(runtime.unreported(result.usage))
        ranked = [k for k in dict.fromkeys(result.ranked_keys) if k in state.pool]
        unknown = [k for k in result.ranked_keys if k not in state.pool]
        state.trace.add("delegate", state.turn, duration_ms=(time.perf_counter() - t0) * 1000,
                        driver=self.driver.id, n_ranked=len(ranked), unknown_keys=unknown[:20],
                        note=result.note)
        state.emitter.emit(PhaseFinished, turn=state.turn, phase="delegate",
                           duration_ms=(time.perf_counter() - t0) * 1000,
                           summary=PhaseSummary(n_ranked=len(ranked), note=result.note))
        if self.analyzer_decider is not None:
            keys = ranked or [c.hit.key for c in state.pool.candidates()]
            state.usage.add_model(await analyzer.judge_keys(state, keys))
        progress()
        return (runtime.exhausted or StopReason.DELEGATE_DONE), (ranked or None)

    def _finalize(self, state: SearchState, reason: StopReason, top_k: int,
                  order: list[str] | None, mode: str) -> SearchResult:
        w = self.settings.judge_weight
        ranked: list[tuple[Candidate, float]]
        if order:
            cands = [state.pool[k] for k in order]
            if any(c.judged for c in cands):
                scores = state.pool.scores(w)
                ranked = sorted(((c, scores[c.hit.key]) for c in cands),
                                key=lambda cs: (-cs[1], cs[0].hit.key))
            else:
                ranked = [(c, 1.0 / (i + 1)) for i, c in enumerate(cands)]
        else:
            ranked = state.pool.ranked(w)
        hits = [RankedHit(hit=c.hit, score=round(s, 6), p_relevant=c.p_relevant,
                          rationale=c.rationale, judged=c.judged) for c, s in ranked[:top_k]]
        state.trace.add("finalize", state.turn, stop_reason=reason.value, n_hits=len(hits),
                        pool_size=len(state.pool))
        return SearchResult(question=state.question, hits=hits, stop_reason=reason,
                            usage=state.usage, trace=state.trace, mode=mode)

    def _closeables(self) -> list[Any]:
        """Backends, plus embedders, deciders, the driver and the objects they wrap (`inner`) or
        hold (`client`), deduplicated by identity, that expose an async `close()`. Backends own their
        clients and close them themselves, so they are not traversed."""
        seen: dict[int, Any] = {id(b): b for b in self.backends.values()}
        roots: list[Any] = [*(self.embedders.get(i) for i in self.embedders.ids()),
                            self.analyzer_decider, self.controller_decider, self.driver]
        while roots:
            obj = roots.pop()
            if obj is None or id(obj) in seen:
                continue
            seen[id(obj)] = obj
            roots.extend(getattr(obj, attr, None) for attr in ("inner", "client"))
        return [obj for obj in seen.values() if inspect.iscoroutinefunction(getattr(obj, "close", None))]

    async def close(self) -> None:
        await asyncio.gather(*(o.close() for o in self._closeables()), return_exceptions=True)

    async def __aenter__(self) -> Harness:
        await self.setup()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -q tests/test_stream.py`
Expected: `15 passed` in well under a second. If a cancellation test takes about 5 seconds and then fails, the stream is not cancelling its search task.

Run: `uv run pytest -q && uv run ruff check`
Expected: `378 passed, 5 skipped, 79 deselected` and `All checks passed!`.

- [ ] **Step 5: Commit**

```bash
git add src/agentic_search/core/harness.py tests/test_stream.py
git commit -m "feat(harness): stream() yields typed search events; search() drains it

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Public exports, README, contract test

**Files:**
- Modify: `src/agentic_search/__init__.py`, `README.md`, `tests/contract/test_backend_contract.py`

**Interfaces:**
- Consumes (Task 3): `SearchStream`, `Harness.stream`.
- Produces: `from agentic_search import SearchEvent, SearchStream, SearchStarted, …` for every event class, plus `HitSummary`, `PhaseSummary` and `ToolErrorInfo`.

- [ ] **Step 1: Write the failing test**

Append to `tests/contract/test_backend_contract.py`. It already imports `Harness`, `Capability`, `KeywordJudge`, `ScriptedDriver`, `call`, `corpus` and `require`.

```python
async def test_stream_matches_search(backend):
    await require(backend, Capability.LEXICAL)

    def build():
        driver = ScriptedDriver([[call("lexical_search", source=backend.name, collection="docs",
                                       text="headache")]])
        return Harness([backend], driver, embedders=[corpus.EMBEDDER],
                       analyzer=KeywordJudge(["headache"]))

    events = [ev async for ev in build().stream("what treats headache?")]
    assert events[0].type == "search_started" and events[-1].type == "search_finished"
    assert [e.seq for e in events] == list(range(len(events)))
    direct = await build().search("what treats headache?")
    assert events[-1].result.keys() == direct.keys()
```

Also append this export test to `tests/test_events.py`:
```python
def test_public_exports():
    import agentic_search as a

    for name in ["SearchEvent", "SearchStream", "SearchStarted", "PhaseStarted", "PhaseFinished",
                 "PhaseSummary", "ToolCallStarted", "ToolCallFinished", "ToolErrorInfo",
                 "ResultsUpdated", "HitSummary", "UsageUpdated", "SearchFinished", "SearchFailed"]:
        assert name in a.__all__ and hasattr(a, name)
```

- [ ] **Step 2: Run the tests to verify the export test fails**

Run: `uv run pytest -q tests/test_events.py::test_public_exports tests/contract -k "exports or stream"`
Expected: `test_public_exports` fails (`assert 'SearchEvent' in [...]`). `test_stream_matches_search[files]` already passes after Task 3; it is the contract check that Task 3 behaves the same on every backend.

- [ ] **Step 3: Replace `src/agentic_search/__init__.py`**

```python
"""Agentic search harness."""

from agentic_search.core.harness import (
           Harness,
           HarnessError,
           HarnessSettings,
           RankedHit,
           SearchResult,
           SearchStream,
)
from agentic_search.core.types import Budget, Query
from agentic_search.events import (
           HitSummary,
           PhaseFinished,
           PhaseStarted,
           PhaseSummary,
           ResultsUpdated,
           SearchEvent,
           SearchFailed,
           SearchFinished,
           SearchStarted,
           ToolCallFinished,
           ToolCallStarted,
           ToolErrorInfo,
           UsageUpdated,
)

__version__ = "0.1.0"

__all__ = ["Budget", "Harness", "HarnessError", "HarnessSettings", "HitSummary", "PhaseFinished",
           "PhaseStarted", "PhaseSummary", "Query", "RankedHit", "ResultsUpdated", "SearchEvent",
           "SearchFailed", "SearchFinished", "SearchResult", "SearchStarted", "SearchStream",
           "ToolCallFinished", "ToolCallStarted", "ToolErrorInfo", "UsageUpdated", "__version__"]
```

- [ ] **Step 4: Add a Streaming section to `README.md`**

Insert this section directly above the `## Modes` heading:

````markdown
## Streaming

`Harness.stream()` runs one search and yields typed events as it goes, so an application can
show progress (planning, querying each source, judging, deciding) and a result list that grows
turn by turn:

```python
from agentic_search import PhaseStarted, ResultsUpdated, SearchFailed, SearchFinished

async with load_harness("search.yaml") as h:
    async with h.stream("what treats headache?", snapshot_k=10) as events:
        async for ev in events:
            match ev:
                case PhaseStarted(phase=phase):
                    print("…", phase)
                case ResultsUpdated(hits=hits):
                    print([s.key for s in hits])
                case SearchFinished(result=result):
                    print(result.keys())
                case SearchFailed(error_type=kind, message=msg):
                    print("failed:", kind, msg)
```

- Every event has `type`, `schema_version`, `search_id`, `seq` (0, 1, 2, … with no gaps),
  `turn` and `at_ms`. Events are Pydantic models; `ev.model_dump_json()` is ready for SSE.
- The last event is always `search_finished` or `search_failed`. Invalid arguments raise
  `HarnessError` from `stream()` itself.
- Leaving the `async with` block or cancelling the consuming task cancels the search, including
  in-flight backend and model calls.
- Snapshots carry a scrubbed 300-character snippet per hit; pass `include_content=True` for full
  content, or `snapshot_k=0` to turn them off.
- `search()` is `stream()` drained to its final result. Consumers must ignore unknown event types.
````

- [ ] **Step 5: Run the full verification**

Run: `uv run pytest -q && uv run ruff check`
Expected: `380 passed, 5 skipped, 84 deselected` and `All checks passed!`.

With Docker services up (`docker compose up -d --wait`):
Run: `AGENTIC_SEARCH_INTEGRATION=1 uv run pytest -q tests/contract -m integration -k stream`
Expected: `5 passed` (postgres, mysql, opensearch, neo4j, milvus).

- [ ] **Step 6: Commit**

```bash
git add src/agentic_search/__init__.py README.md tests/contract/test_backend_contract.py tests/test_events.py
git commit -m "feat: export streaming events; README streaming section; contract stream test

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Spec coverage

| Spec section | Task |
|---|---|
| §2 API: `stream`, `SearchStream`, `search` on top, validation | 3 (exports: 4) |
| §3.1–3.2 common fields, event types, `PhaseSummary`, `ToolErrorInfo`, `HitSummary` | 1 |
| §3.3 ordering (gap-free `seq`, paired phases and calls, single terminal event) | 1 (`seq`), 2 (pairs), 3 (terminal, end-to-end tests) |
| §3.4 per-mode sequences | 3 (`test_retrieval/harness/model_mode_sequence`) |
| §3.5 snapshots equal the final head (retrieval/harness) | 3 (`test_harness_mode_sequence_and_final_snapshot`) |
| §3.6 versioning, ignoring unknown types | 1 (`schema_version`), 4 (README) |
| §4.1 emitter, `NullEmitter` default | 1 |
| §4.2 emission points | 2, 3 |
| §4.3 queue, task, `search()` re-raises the original | 3 |
| §4.4 cancellation | 3 (two cancellation tests) |
| §4.5 concurrent streams, setup once | 3 |
| §4.6 scrubbing, `include_content` | 1, 2, 3 |
| §5 error handling table | 3 (failure tests) |
| §6 testing items 1–11 | 1–4 (item 11: every task's full-suite run) |
| §6 integration | 4 |

