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
from agentic_search.models.base import Action, DelegateResult
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
    opened = [(e.phase, e.turn) for e in events if isinstance(e, PhaseStarted)]
    closed = [(e.phase, e.turn) for e in events if isinstance(e, PhaseFinished)]
    assert opened == closed
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


async def test_cancelled_error_from_driver_plan_search_propagates(docs_backend):
    """Test that CancelledError raised inside plan() is not swallowed by search()."""
    class CancellingDriver(ScriptedDriver):
        async def plan(self, view, tools):
            raise asyncio.CancelledError("plan cancelled")

    h = make(docs_backend, CancellingDriver())
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(h.search("q"), 5)


async def test_cancelled_error_from_driver_plan_stream_propagates(docs_backend):
    """Test that CancelledError raised inside plan() is not swallowed by stream()."""
    class CancellingDriver(ScriptedDriver):
        async def plan(self, view, tools):
            raise asyncio.CancelledError("plan cancelled")

    h = make(docs_backend, CancellingDriver())
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(collect(h.stream("q")), 5)


async def test_custom_base_exception_from_driver_search_propagates(docs_backend):
    """Test that custom BaseException subclass is not swallowed by search()."""
    class Fatal(BaseException):
        pass

    class FatalDriver(ScriptedDriver):
        async def plan(self, view, tools):
            raise Fatal("custom fatal error")

    h = make(docs_backend, FatalDriver())
    with pytest.raises(Fatal):
        await asyncio.wait_for(h.search("q"), 5)


async def test_custom_base_exception_from_driver_stream_propagates(docs_backend):
    """Test that custom BaseException subclass is not swallowed by stream()."""
    class Fatal(BaseException):
        pass

    class FatalDriver(ScriptedDriver):
        async def plan(self, view, tools):
            raise Fatal("custom fatal error")

    h = make(docs_backend, FatalDriver())
    with pytest.raises(Fatal):
        await asyncio.wait_for(collect(h.stream("q")), 5)


async def test_consumer_timeout_not_swallowed_by_stream_cleanup(docs_backend):
    """Test that consumer's timeout fires even during stream cleanup.

    The timeout must expire while _events' finally block awaits the search task
    during its shielded cleanup. Without the fix, contextlib.suppress swallows
    the consumer's timeout and the test hangs.
    """
    async def long_sleep_with_shielded_cleanup(op):
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            # Backend does shielded cleanup before re-raising
            await asyncio.shield(asyncio.sleep(0.3))
            raise
        return []

    docs_backend.execute = long_sleep_with_shielded_cleanup
    h = make(docs_backend, ScriptedDriver([[lex("headache")]]),
             settings=HarnessSettings(call_timeout=600))

    async def consume_with_timeout():
        async with asyncio.timeout(0.1):
            async with h.stream("q") as s:
                async for ev in s:
                    if isinstance(ev, ToolCallStarted):
                        break  # Exit loop immediately; aclose() cancels search task

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(consume_with_timeout(), 5)


async def test_delegate_note_secret_is_scrubbed(docs_backend):
    register_secret("sk-note-4242")

    class NotingDriver(ScriptedDriver):
        async def run_delegate(self, question, tools, runtime, budget, context=""):
            result = await super().run_delegate(question, tools, runtime, budget, context=context)
            return DelegateResult(ranked_keys=result.ranked_keys, usage=result.usage,
                                  note="used key sk-note-4242")

    h = make(docs_backend, NotingDriver(delegate_calls=[[lex("headache")]], delegate_keys=["docs:d1"]))
    events = await collect(h.stream("q", mode="model"))
    done = [e for e in events if isinstance(e, PhaseFinished) and e.phase == "delegate"][0]
    assert done.summary.note is not None
    assert all("sk-note-4242" not in e.model_dump_json() for e in events[:-1])
