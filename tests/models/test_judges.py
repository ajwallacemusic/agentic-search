import pytest

from agentic_search import Harness
from agentic_search.core.types import Hit, Query, TextPart
from agentic_search.models.base import Action, ControllerView, ToolCall, TurnSummary
from agentic_search.models.cross_encoder import CrossEncoderJudge
from agentic_search.models.driver import ToolCallingDriver
from agentic_search.models.llm import ChatResponse
from agentic_search.models.llm_judge import LLMJudge
from agentic_search.testing import FakeLLMClient

HITS = [Hit(doc_id="1", source="s", content=[TextPart(text="aspirin for headache")]),
        Hit(doc_id="2", source="s", content=[TextPart(text="castles")])]


def submit(name, **args):
    return ChatResponse(tool_calls=[ToolCall(id="x", name=name, arguments=args)])


async def test_llm_judge_maps_indices_and_clamps():
    client = FakeLLMClient([submit("submit_judgments", judgments=[
        {"index": 0, "p_relevant": 1.4, "rationale": "on topic"},
        {"index": 1, "p_relevant": 0.1},
        {"index": 9, "p_relevant": 0.5},
        {"index": "bad"},
    ])])
    j = LLMJudge(client)
    assert j.id == "llm-judge:fake-llm"
    res = await j.judge(Query.of("headache"), HITS)
    assert [(x.key, x.p_relevant) for x in res.judgments] == [("s:1", 1.0), ("s:2", 0.1)]
    req = client.requests[0]
    assert req["tool_choice"] == "submit_judgments"
    assert "[0] s:1" in req["messages"][0].content[0].text


async def test_llm_judge_requires_tool_call():
    with pytest.raises(ValueError):
        await LLMJudge(FakeLLMClient([ChatResponse(text="no")])).judge(Query.of("q"), HITS)
    assert (await LLMJudge(FakeLLMClient([])).judge(Query.of("q"), [])).judgments == []


async def test_llm_judge_decide():
    client = FakeLLMClient([submit("submit_decision", action="broaden", confidence=0.7, note="n")])
    view = ControllerView(question=Query.of("q"), turn=1, digest="d", total_relevant=0,
                          history=[TurnSummary(turn=0, n_calls=2, n_errors=0, n_new=3,
                                               n_new_relevant=0)],
                          budget_remaining={"turns": 2})
    d = await LLMJudge(client).decide(view)
    assert d.action is Action.BROADEN and d.confidence == 0.7
    assert "turn 0: 2 calls" in client.requests[0]["messages"][0].content


async def test_cross_encoder_with_injected_scorer():
    seen = []

    def scorer(pairs):
        seen.extend(pairs)
        return [2.0, -2.0]

    j = CrossEncoderJudge("m", scorer=scorer, apply_sigmoid=True)
    res = await j.judge(Query.of("headache"), HITS)
    assert seen[0] == ("headache", "aspirin for headache")
    assert res.judgments[0].p_relevant == pytest.approx(0.8808, abs=1e-3)
    assert res.judgments[1].p_relevant == pytest.approx(0.1192, abs=1e-3)
    with pytest.raises(NotImplementedError):
        await j.decide(None)
    clamp = CrossEncoderJudge("m", scorer=lambda p: [1.7, -0.2])
    assert [x.p_relevant for x in (await clamp.judge(Query.of("q"), HITS)).judgments] == [1.0, 0.0]


async def test_end_to_end_with_llm_driver_and_judge(docs_backend):
    driver_client = FakeLLMClient([
        ChatResponse(tool_calls=[ToolCall(id="1", name="lexical_search",
                                          arguments={"source": "docs", "text": "headache"})]),
        ChatResponse(text="done"),
    ])
    judge_client = FakeLLMClient([submit("submit_judgments", judgments=[
        {"index": 0, "p_relevant": 0.9}, {"index": 1, "p_relevant": 0.8}])], id="judge")
    h = Harness([docs_backend], ToolCallingDriver(driver_client), analyzer=LLMJudge(judge_client))
    res = await h.search("what treats headache?")
    assert set(res.keys()) == {"docs:d1", "docs:d4"} and all(r.judged for r in res.hits)


async def test_cross_encoder_lazy_load_happens_once_under_concurrency(monkeypatch):
    import asyncio
    import time

    loads = []

    def fake_default_scorer(self):
        loads.append(1)
        time.sleep(0.05)  # slow model load, run in a worker thread
        return lambda pairs: [0.5] * len(pairs)

    monkeypatch.setattr(CrossEncoderJudge, "_default_scorer", fake_default_scorer)
    j = CrossEncoderJudge("m")
    r1, r2 = await asyncio.gather(j.judge(Query.of("q"), HITS), j.judge(Query.of("q"), HITS))
    assert len(loads) == 1
    assert [x.p_relevant for x in r1.judgments] == [0.5, 0.5] == [x.p_relevant for x in r2.judgments]
