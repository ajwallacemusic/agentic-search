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
