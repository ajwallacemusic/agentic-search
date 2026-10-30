import pytest
from pydantic import TypeAdapter, ValidationError

from agentic_search.core.emitter import EventEmitter, NullEmitter
from agentic_search.core.result import RankedHit, SearchResult
from agentic_search.core.secrets import register_secret
from agentic_search.core.state import SearchState, Trace, Usage
from agentic_search.core.types import Budget, Hit, Query, StopReason, TextPart
from agentic_search.events import (
    SCHEMA_VERSION,
    FactNoted,
    HitSummary,
    PhaseFinished,
    PhaseStarted,
    PhaseSummary,
    ResultsUpdated,
    SearchEvent,
    SearchFailed,
    SearchFinished,
    SearchStarted,
    StepsUpdated,
    ToolCallFinished,
    ToolCallStarted,
    ToolErrorInfo,
    UsageUpdated,
)
from agentic_search.models.base import Step

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
        StepsUpdated(**COMMON, steps=[Step(text="a", done=True), Step(text="b")], current=1),
        FactNoted(**COMMON, fact="f", source_key="docs:d1"),
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
    assert len({e.type for e in one_of_each()}) == 11


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


def test_public_exports():
    import agentic_search as a

    for name in ["SearchEvent", "SearchStream", "SearchStarted", "PhaseStarted", "PhaseFinished",
                 "PhaseSummary", "ToolCallStarted", "ToolCallFinished", "ToolErrorInfo",
                 "ResultsUpdated", "HitSummary", "UsageUpdated", "SearchFinished", "SearchFailed", "StepsUpdated", "FactNoted"]:
        assert name in a.__all__ and hasattr(a, name)


def _literal_values(annotation):
    from typing import Literal, Union, get_args, get_origin
    if get_origin(annotation) is Literal:
        return set(get_args(annotation))
    assert get_origin(annotation) in (Union, type(int | str)), annotation
    [lit] = [a for a in get_args(annotation) if get_origin(a) is Literal]
    return set(get_args(lit))


def test_narrowed_literals_match_their_sources():
    from agentic_search.core.types import ToolError
    from agentic_search.models.base import Action
    assert _literal_values(PhaseSummary.model_fields["action"].annotation) == {a.value for a in Action}
    assert (_literal_values(ToolErrorInfo.model_fields["kind"].annotation)
            == _literal_values(ToolError.model_fields["kind"].annotation))
    assert _literal_values(SearchStarted.model_fields["mode"].annotation) == {
        "retrieval", "harness", "model"}


def test_narrowed_literals_reject_unknown_values():
    with pytest.raises(ValidationError):
        ToolErrorInfo(kind="weird", message="m")
    with pytest.raises(ValidationError):
        PhaseSummary(action="explode")


def _steps_harness(docs_backend, turns, steps=True):
    from agentic_search import Harness, HarnessSettings
    from agentic_search.testing import ScriptedDriver
    return Harness([docs_backend], ScriptedDriver(turns), embedders=[docs_backend.embedder],
                   settings=HarnessSettings(steps=steps))


async def _events(h):
    return [ev async for ev in h.stream("q", budget=Budget(max_turns=3))]


async def test_steps_and_facts_stream_as_events(docs_backend):
    from agentic_search.testing import call

    def lex(text):
        return call("lexical_search", source="docs", text=text)

    h = _steps_harness(docs_backend, [
        [call("plan_steps", steps=["a", "b"]), lex("headache")],
        [call("note", fact="f1", source_key="docs:d1"), call("note", fact="bad", source_key="x:y"),
         call("step_done", step=1), lex("fever")]])
    events = await _events(h)
    steps = [e for e in events if e.type == "steps_updated"]
    facts = [e for e in events if e.type == "fact_noted"]
    assert [(e.turn, [(s.text, s.done) for s in e.steps], e.current) for e in steps] == [
        (0, [("a", False), ("b", False)], 0), (1, [("a", True), ("b", False)], 1)]
    assert [(e.turn, e.fact, e.source_key) for e in facts] == [(1, "f1", "docs:d1")]
    for e in steps + facts:  # SSE round trip
        back = ADAPTER.validate_json(e.model_dump_json())
        assert type(back) is type(e) and back == e


async def test_note_alone_does_not_emit_steps_updated(docs_backend):
    from agentic_search.testing import call
    h = _steps_harness(docs_backend, [
        [call("lexical_search", source="docs", text="headache")],
        [call("note", fact="f1", source_key="docs:d1"),
         call("lexical_search", source="docs", text="fever")]])
    events = await _events(h)
    assert not [e for e in events if e.type == "steps_updated"]
    assert [e.type for e in events if e.type == "fact_noted"] == ["fact_noted"]


async def test_no_step_events_when_off(docs_backend):
    from agentic_search.testing import call
    h = _steps_harness(docs_backend, [
        [call("plan_steps", steps=["a"]), call("lexical_search", source="docs", text="headache")]],
        steps=False)
    events = await _events(h)
    assert not [e for e in events if e.type in ("steps_updated", "fact_noted")]
