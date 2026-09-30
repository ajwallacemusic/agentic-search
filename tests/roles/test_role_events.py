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
