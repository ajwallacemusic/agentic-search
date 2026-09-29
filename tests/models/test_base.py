import pytest
from pydantic import ValidationError

from agentic_search.core.types import Budget, Hit, Query, TextPart
from agentic_search.models.base import (
    Action, Decider, Decision, Driver, Judgment, PlannerView,
)
from agentic_search.testing import (
    EchoDriver, FailingDecider, KeywordJudge, ScriptedController, ScriptedDriver, call,
)


def view(turn=0):
    return PlannerView(question=Query.of("q"), turn=turn, manifest_summary="", digest="")


def test_value_bounds():
    with pytest.raises(ValidationError):
        Judgment(key="k", p_relevant=1.5)
    with pytest.raises(ValidationError):
        Decision(action=Action.STOP, confidence=-0.1)


def test_call_helper_assigns_ids():
    a, b = call("lexical_search", text="x"), call("lexical_search", text="y")
    assert a.id != b.id and a.arguments == {"text": "x"}


async def test_scripted_driver():
    d = ScriptedDriver([[call("x")]])
    assert isinstance(d, Driver)
    assert len((await d.plan(view(), [])).calls) == 1
    assert (await d.plan(view(1), [])).calls == []
    assert len(d.views) == 2


async def test_scripted_driver_delegate():
    class Runtime:
        async def call(self, calls):
            return [f"out:{c.name}" for c in calls]

    d = ScriptedDriver(delegate_calls=[[call("a")]], delegate_keys=["s:1"])
    res = await d.run_delegate(Query.of("q"), [], Runtime(), Budget(), context="ctx")
    assert res.ranked_keys == ["s:1"] and d.delegate_outputs == [["out:a"]]
    assert d.delegate_context == "ctx"


async def test_echo_driver():
    d = EchoDriver("docs", limit=5)
    [c] = (await d.plan(view(), [])).calls
    assert c.name == "lexical_search" and c.arguments == {"source": "docs", "text": "q", "limit": 5}
    assert (await d.plan(view(1), [])).calls == []


async def test_keyword_judge_and_controllers():
    j = KeywordJudge(["fever"])
    assert isinstance(j, Decider)
    hits = [Hit(doc_id="1", source="s", content=[TextPart(text="Fever high")]),
            Hit(doc_id="2", source="s", content=[TextPart(text="nothing")])]
    res = await j.judge(Query.of("q"), hits)
    assert [x.p_relevant for x in res.judgments] == [1.0, 0.0]
    with pytest.raises(NotImplementedError):
        await j.decide(None)
    c = ScriptedController([Action.CONTINUE])
    assert (await c.decide(None)).action is Action.CONTINUE
    assert (await c.decide(None)).action is Action.STOP
    with pytest.raises(RuntimeError):
        await FailingDecider().judge(Query.of("q"), hits)
