"""Decision 1 judge against httpx.MockTransport (offline)."""

import json

import httpx
import pytest

from agentic_search.core.types import Hit, Query, TextPart
from demo.ignite.decision_judge import DecisionJudge


def _hits():
    return [Hit(doc_id=str(i), source="products", content=[TextPart(text=t)])
            for i, t in enumerate(["teak patio table", "bath faucet"])]


def _judge(handler):
    return DecisionJudge(key="k", price_per_mtok_in=0.10,
                         client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def test_one_noul_question_per_hit_and_cost_from_input_tokens():
    seen = {}

    def handler(req):
        seen["body"], seen["key"] = json.loads(req.content), req.headers["x-goog-api-key"]
        return httpx.Response(200, json={
            "answers": {"d0": {"type": "noul", "noul": 0.97}, "d1": {"type": "noul", "noul": 0.02}},
            "usage": {"total_input_tokens": 1000, "total_output_tokens": 0}})

    res = await _judge(handler).judge(Query.of("light wood patio table"), _hits())
    body = seen["body"]
    assert seen["key"] == "k" and body["model"] == "gemini-decision-1"
    assert set(body["questions"]) == {"d0", "d1"}
    assert all(q["type"] == "noul" for q in body["questions"].values())
    assert "light wood patio table" in body["input"] and "[1] products:1" in body["input"]
    assert [(j.key, j.p_relevant) for j in res.judgments] == [("products:0", 0.97),
                                                              ("products:1", 0.02)]
    assert res.usage.input_tokens == 1000 and res.usage.output_tokens == 0
    assert res.usage.cost_usd == pytest.approx(0.0001)


async def test_missing_answer_is_skipped_not_guessed():
    def handler(req):
        return httpx.Response(200, json={"answers": {"d0": {"noul": 0.6}}, "usage": {}})

    res = await _judge(handler).judge(Query.of("q"), _hits())
    assert [j.key for j in res.judgments] == ["products:0"]


async def test_retries_server_errors_then_succeeds():
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"answers": {"d0": {"noul": 0.5}, "d1": {"noul": 0.5}},
                                         "usage": {"total_input_tokens": 10}})

    res = await _judge(handler).judge(Query.of("q"), _hits())
    assert len(calls) == 2 and len(res.judgments) == 2


async def test_bad_request_raises_without_retry():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(400, json={"error": {"message": "bad type"}})

    with pytest.raises(ValueError, match="400"):
        await _judge(handler).judge(Query.of("q"), _hits())
    assert len(calls) == 1


async def test_no_hits_makes_no_call():
    def handler(req):
        raise AssertionError("no call expected")

    assert (await _judge(handler).judge(Query.of("q"), [])).judgments == []
