"""TypeSafe decider against httpx.MockTransport; `test_live_typesafe` hits the real API when
TYPESAFE_API_KEY is set."""

import json
import os

import httpx
import pytest

from agentic_search.core.secrets import scrub
from agentic_search.core.types import Hit, Query, TextPart
from agentic_search.models.base import Action, ControllerView, Decider, TurnSummary
from agentic_search.models.typesafe import TypeSafeDecider, TypeSafeError

HITS = [Hit(doc_id=str(i), source="s", content=[TextPart(text=t)])
        for i, t in enumerate(["aspirin relieves headache", "medieval castles", "ibuprofen for pain"])]


def client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_judge_batches_nouls_over_shared_state():
    seen = []

    def respond(req):
        body = json.loads(req.content)
        seen.append((req, body))
        answers = {qid: {"type": "noul", "noul": 0.9 if "headache" in body["state"]["documents"][int(qid[1:])]["text"] else 0.1}
                   for qid in body["questions"]}
        return httpx.Response(200, json={"model": "jev-latest", "answers": answers,
                                         "usage": {"input_tokens": 100, "output_tokens": 3}})

    d = TypeSafeDecider(api_key="ts-key-12345", client=client(respond), batch_size=2,
                        price_per_mtok=(1.0, 1.0))
    assert isinstance(d, Decider) and d.id == "typesafe:jev-latest"
    res = await d.judge(Query.of("what treats headache?"), HITS)
    assert [(j.key, j.p_relevant) for j in res.judgments] == [("s:0", 0.9), ("s:1", 0.1), ("s:2", 0.1)]
    assert len(seen) == 2 and res.usage.input_tokens == 200 and res.usage.cost_usd == pytest.approx(206e-6)
    req, body = seen[0]
    assert str(req.url) == "https://api.typesafe.ai/v1/systemone"
    assert req.headers["authorization"] == "Bearer ts-key-12345"
    assert body["model"] == "jev-latest" and body["state"]["query"] == "what treats headache?"
    assert body["questions"]["d1"]["type"] == "noul"
    assert "`documents[1].text`" in body["questions"]["d1"]["instructions"]
    assert set(body["questions"]["d0"]["criteria"]) == {"true", "false"}


async def test_decide_uses_choice():
    def respond(req):
        body = json.loads(req.content)
        assert body["questions"]["action"]["type"] == "choice"
        assert set(body["questions"]["action"]["criteria"]) == {a.value for a in Action}
        return httpx.Response(200, json={"answers": {"action": {"type": "choice", "choice": "broaden",
                                                                 "confidence": 0.8}}})

    view = ControllerView(question=Query.of("q"), turn=1, digest="d", total_relevant=0,
                          history=[TurnSummary(turn=0, n_calls=2, n_errors=0, n_new=3, n_new_relevant=0)],
                          budget_remaining={"turns": 2})
    d = await TypeSafeDecider(api_key="ts-key-12345", client=client(respond)).decide(view)
    assert d.action is Action.BROADEN and d.confidence == 0.8


async def test_errors_and_retries():
    calls = {"n": 0}

    def flaky(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(529, text="overloaded")
        return httpx.Response(200, json={"answers": {"d0": {"noul": 0.5}}})

    d = TypeSafeDecider(api_key="ts-key-12345", client=client(flaky), backoff_s=0.001)
    assert (await d.judge(Query.of("q"), HITS[:1])).judgments[0].p_relevant == 0.5
    assert calls["n"] == 2

    bad = TypeSafeDecider(api_key="ts-key-12345", client=client(lambda r: httpx.Response(
        401, text="invalid key ts-key-12345")), backoff_s=0.001)
    with pytest.raises(TypeSafeError) as exc:
        await bad.judge(Query.of("q"), HITS[:1])
    assert "401" in str(exc.value) and "ts-key-12345" not in str(exc.value)

    no_choice = TypeSafeDecider(api_key="ts-key-12345", client=client(lambda r: httpx.Response(
        200, json={"answers": {}})))
    view = ControllerView(question=Query.of("q"), turn=0, digest="", total_relevant=None, history=[],
                          budget_remaining={})
    with pytest.raises(TypeSafeError, match="no usable choice"):
        await no_choice.decide(view)


async def test_missing_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(TypeSafeError, match="API key"):
        await TypeSafeDecider(client=client(lambda r: httpx.Response(200))).judge(Query.of("q"), HITS[:1])


async def test_env_key_registered(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-env-key-777")
    TypeSafeDecider()
    assert scrub("ts-env-key-777") == "***"


async def test_malformed_2xx_response():
    """2xx response with non-JSON body should raise TypeSafeError."""
    def respond(req):
        return httpx.Response(200, text="<html>Internal Server Error</html>")

    d = TypeSafeDecider(api_key="ts-key-12345", client=client(respond))
    with pytest.raises(TypeSafeError, match="malformed response"):
        await d.judge(Query.of("q"), HITS[:1])


async def test_non_numeric_confidence_in_decide():
    """Non-numeric confidence should default to 1.0 in decide."""
    def respond(req):
        return httpx.Response(200, json={"answers": {"action": {"type": "choice", "choice": "broaden",
                                                                 "confidence": "high"}}})

    view = ControllerView(question=Query.of("q"), turn=0, digest="", total_relevant=None, history=[],
                          budget_remaining={})
    d = TypeSafeDecider(api_key="ts-key-12345", client=client(respond))
    result = await d.decide(view)
    assert result.action is Action.BROADEN and result.confidence == 1.0


async def test_partial_batch_results():
    """If second batch fails, first batch judgments should be returned."""
    calls = {"n": 0}

    def flaky(req):
        calls["n"] += 1
        if calls["n"] == 1:
            # First batch succeeds
            answers = {"d0": {"noul": 0.9}, "d1": {"noul": 0.1}}
            return httpx.Response(200, json={"answers": answers,
                                             "usage": {"input_tokens": 100, "output_tokens": 3}})
        else:
            # Second batch always fails
            return httpx.Response(500, text="server error")

    d = TypeSafeDecider(api_key="ts-key-12345", client=client(flaky), batch_size=2,
                        backoff_s=0.001, max_retries=0)
    res = await d.judge(Query.of("q"), HITS)  # 3 hits, 2 per batch
    # Should have 2 judgments (first batch) not 3
    assert len(res.judgments) == 2
    assert [(j.key, j.p_relevant) for j in res.judgments] == [("s:0", 0.9), ("s:1", 0.1)]
    assert res.usage.input_tokens == 100


async def test_missing_answer_omitted():
    """Hit with missing answer in response should be omitted."""
    def respond(req):
        # Only return answer for d0, not d1
        answers = {"d0": {"noul": 0.8}}
        return httpx.Response(200, json={"answers": answers,
                                         "usage": {"input_tokens": 100, "output_tokens": 3}})

    d = TypeSafeDecider(api_key="ts-key-12345", client=client(respond), batch_size=2)
    res = await d.judge(Query.of("q"), HITS[:2])
    assert len(res.judgments) == 1
    assert res.judgments[0].key == "s:0" and res.judgments[0].p_relevant == 0.8


async def test_401_no_retry():
    """401 error should not retry (non-RETRY_STATUSES)."""
    calls = {"n": 0}

    def respond(req):
        calls["n"] += 1
        return httpx.Response(401, text="unauthorized")

    d = TypeSafeDecider(api_key="ts-key-12345", client=client(respond), backoff_s=0.001)
    with pytest.raises(TypeSafeError):
        await d.judge(Query.of("q"), HITS[:1])
    assert calls["n"] == 1  # Should make exactly one call, no retries


@pytest.mark.live
async def test_live_typesafe():
    if not os.environ.get("TYPESAFE_API_KEY"):
        pytest.skip("set TYPESAFE_API_KEY")
    d = TypeSafeDecider()
    try:
        res = await d.judge(Query.of("what relieves headaches?"), HITS)
    finally:
        await d.close()
    p = {j.key: j.p_relevant for j in res.judgments}
    assert p["s:0"] > p["s:1"]


async def test_secret_straddling_truncation_point_is_not_leaked():
    from agentic_search.core.secrets import register_secret
    secret = "ts-straddle-secret-98765"
    register_secret(secret)
    body = "x" * 290 + secret
    d = TypeSafeDecider(api_key="ts-key-12345", client=client(lambda r: httpx.Response(400, text=body)))
    with pytest.raises(TypeSafeError) as exc:
        await d.judge(Query.of("q"), HITS[:1])
    assert "ts-straddl" not in str(exc.value)


async def test_non_dict_answers_is_typesafe_error_with_usage():
    usage = {"input_tokens": 50, "output_tokens": 2}
    d = TypeSafeDecider(api_key="ts-key-12345",
                        client=client(lambda r: httpx.Response(200, json={"answers": [1, 2], "usage": usage})))
    with pytest.raises(TypeSafeError, match="answers") as exc:
        await d.judge(Query.of("q"), HITS[:1])
    assert exc.value.usage.input_tokens == 50
    view = ControllerView(question=Query.of("q"), turn=0, digest="", total_relevant=None, history=[],
                          budget_remaining={})
    with pytest.raises(TypeSafeError, match="answers"):
        await d.decide(view)


async def test_partial_batch_keeps_usage_of_malformed_paid_batch():
    calls = {"n": 0}

    def respond(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json={"answers": {"d0": {"noul": 0.9}, "d1": {"noul": 0.2}},
                                             "usage": {"input_tokens": 100, "output_tokens": 3}})
        return httpx.Response(200, json={"answers": "oops", "usage": {"input_tokens": 40, "output_tokens": 1}})

    d = TypeSafeDecider(api_key="ts-key-12345", client=client(respond), batch_size=2)
    res = await d.judge(Query.of("q"), HITS)
    assert len(res.judgments) == 2
    assert res.usage.input_tokens == 140


async def test_non_dict_answer_entries_are_treated_as_missing():
    answers = {"d0": [0.9], "d1": {"noul": 0.3}, "action": "stop"}

    d = TypeSafeDecider(api_key="ts-key-12345",
                        client=client(lambda r: httpx.Response(200, json={"answers": answers})), batch_size=2)
    res = await d.judge(Query.of("q"), HITS[:2])
    assert [(j.key, j.p_relevant) for j in res.judgments] == [("s:1", 0.3)]
    view = ControllerView(question=Query.of("q"), turn=0, digest="", total_relevant=None, history=[],
                          budget_remaining={})
    with pytest.raises(TypeSafeError, match="no usable choice"):
        await d.decide(view)
