import asyncio
import contextlib
import json
from dataclasses import dataclass
from typing import Any

import pytest

from agentic_search.core.types import Budget, ModelUsage
from agentic_search.embedders.local import HashEmbedder
from agentic_search.events import ResultsUpdated, SearchFailed
from agentic_search.models.llm import ChatResponse
from agentic_search.testing import KeywordJudge, ScriptedDriver, call
from demo.ignite.build import build_demo_harness, build_demo_swarm, load_config
from demo.ignite.swarm import (
    Swarm,
    SwarmConfig,
    agent_budget,
    agent_query,
    parse_split,
    rrf,
    split_question,
)
from demo.ignite.tests.helpers import HERO_ID, demo_data


class RaisingClient:
    id = "raising"

    async def chat(self, system, messages, **kw):
        raise TimeoutError("slow")


def test_parse_split_reads_fenced_json():
    subs, fb = parse_split('```json\n{"sub_questions": ["a?", "b?", "c?"]}\n```', "Q", 4)
    assert subs == ["a?", "b?", "c?"] and fb is False


def test_parse_split_caps_and_dedupes():
    subs, _ = parse_split('{"sub_questions": ["a","a","b","c","d","e","f"]}', "Q", 4)
    assert subs == ["a", "b", "c", "d"]


@pytest.mark.parametrize("text", ["no json here", '{"sub_questions": []}', '{"sub_questions": ["only"]}'])
def test_parse_split_falls_back_to_the_question(text):
    assert parse_split(text, "Q", 4) == (["Q"], True)


async def test_split_question_falls_back_when_the_model_errors():
    split = await split_question(RaisingClient(), "Q", {"wiki": "d"}, 4)
    assert split.sub_questions == ["Q"] and split.fallback and split.usage.cost_usd == 0


def test_agent_budget_divides_the_harness_budget():
    total = Budget(max_turns=4, max_tool_calls=24, max_seconds=120, max_cost_usd=0.50)
    b = agent_budget(total, 4, split_cost=0.02, max_turns=2)
    assert (b.max_turns, b.max_tool_calls, b.max_seconds) == (2, 6, 120)
    assert b.max_cost_usd == pytest.approx(0.12)


def test_agent_budget_gives_at_least_one_call():
    assert agent_budget(Budget(max_tool_calls=3), 6, 0.0, 2).max_tool_calls == 1


def test_agent_budget_keeps_unlimited_cost_unlimited():
    assert agent_budget(Budget(max_cost_usd=None), 4, 0.02, 2).max_cost_usd is None


def test_agent_query_text():
    assert agent_query("Q", "P") == "Main question: Q. Your part: P."


def test_rrf_sums_reciprocal_ranks():
    out = rrf([["a", "b"], ["b", "c"]], k=60)
    assert [key for key, _, _ in out] == ["b", "a", "c"]
    assert out[0][1] == pytest.approx(1/62 + 1/61) and out[0][2] == [0, 1]


def test_rrf_ties_keep_first_seen_order():
    assert [k for k, _, _ in rrf([["x"], ["y"]], k=60)] == ["x", "y"]


# --- Swarm: fan-out stream and search (offline) ---

class FakeSplitClient:
    """Answers the split call with fixed sub-questions, at a cost of $0.01."""

    id = "fake-split"

    def __init__(self, subs: list[str]):
        self.subs = subs

    async def chat(self, system, messages, **kw):
        return ChatResponse(text=json.dumps({"sub_questions": self.subs}),
                            usage=ModelUsage(input_tokens=100, output_tokens=20, cost_usd=0.01))


def swarm_harness():
    """Offline harness whose planner searches the hero paragraphs on every turn of every agent."""
    driver = ScriptedDriver([[call("lexical_search", source="wiki", text=f"{HERO_ID} text {i % 3}",
                                   limit=10)] for i in range(40)])
    return build_demo_harness(load_config(), data=demo_data(), driver=driver,
                              analyzer=KeywordJudge([HERO_ID]), embedder=HashEmbedder())


@dataclass
class StreamCall:
    question: str
    budget: Budget


class SpyHarness:
    """Wraps a harness: records each stream call; agent `fail_agent` fails at once, agent
    `stall_agent` sends one results_updated and then hangs."""

    def __init__(self, inner, *, fail_agent: int | None = None, stall_agent: int | None = None):
        self.inner, self.fail_agent, self.stall_agent = inner, fail_agent, stall_agent
        self.calls: list[StreamCall] = []
        self.stall_closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def stream(self, question, **kw):
        idx = len(self.calls)
        self.calls.append(StreamCall(question, kw["budget"]))
        if idx == self.fail_agent:
            return self._fail()
        if idx == self.stall_agent:
            return self._stall(question, kw)
        return self.inner.stream(question, **kw)

    async def _fail(self):
        yield SearchFailed(search_id="x", seq=0, turn=0, at_ms=0.0, error_type="RuntimeError",
                           message="model unavailable")

    async def _stall(self, question, kw):
        try:
            async with contextlib.aclosing(self.inner.stream(question, **kw)) as events:
                async for ev in events:
                    if isinstance(ev, ResultsUpdated) and ev.hits:
                        yield ev
                        break
            await asyncio.sleep(3600)
        finally:
            self.stall_closed = True


def make_swarm(*, subs, harness=None, fail_agent=None, stall_agent=None, max_seconds=None):
    harness = harness or SpyHarness(swarm_harness(), fail_agent=fail_agent,
                                    stall_agent=stall_agent)
    if max_seconds is not None:
        harness.inner.budget = harness.inner.budget.model_copy(update={"max_seconds": max_seconds})
    return Swarm(harness, FakeSplitClient(subs), SwarmConfig(), demo_data().descriptions)


async def collect(events):
    return [e async for e in events]


@pytest.fixture
def spy_harness():
    return SpyHarness(swarm_harness())


async def test_stream_event_order_and_merge():
    evs = [e async for e in make_swarm(subs=["a", "b", "c"]).stream("Q")]
    types = [e["type"] for e in evs]
    assert types[0:2] == ["swarm_started", "swarm_split"] and types[-1] == "swarm_finished"
    assert {e["agent"] for e in evs if e["type"] == "agent_event"} == {0, 1, 2}
    fin = evs[-1]
    assert fin["agent_status"] == ["done", "done", "done"] and len(fin["hits"]) <= 10
    assert fin["usage"]["cost_usd"] >= 0.01   # split cost included


async def test_wire_fields_match_the_global_format():
    evs = await collect(make_swarm(subs=["a", "b"]).stream("Q"))
    started, split, fin = evs[0], evs[1], evs[-1]
    assert started["question"] == "Q" and started["max_agents"] == 4
    assert started["budget"]["max_seconds"] == 120 and started["at_ms"] == 0
    assert split["sub_questions"] == ["a", "b"] and split["fallback"] is False
    assert split["usage"] == {"input_tokens": 100, "output_tokens": 20, "cost_usd": 0.01}
    agent_ev = next(e for e in evs if e["type"] == "agent_event")
    assert isinstance(agent_ev["event"], dict) and "type" in agent_ev["event"]
    assert set(fin["usage"]) == {"input_tokens", "output_tokens", "cost_usd", "tool_calls", "turns"}
    hit = fin["hits"][0]
    assert set(hit) == {"key", "source", "doc_id", "title", "snippet", "score", "agents"}
    assert hit["key"] == f"{hit['source']}:{hit['doc_id']}" and hit["agents"]
    assert fin["seconds"] > 0 and fin["at_ms"] >= split["at_ms"]


async def test_each_agent_gets_the_divided_budget_and_its_query(spy_harness):
    [e async for e in make_swarm(subs=["a", "b", "c", "d"], harness=spy_harness).stream("Q")]
    assert {c.budget.max_tool_calls for c in spy_harness.calls} == {6}
    assert spy_harness.calls[0].question == "Main question: Q. Your part: a."


async def test_one_failed_agent_does_not_sink_the_swarm():
    fin = (await collect(make_swarm(subs=["a", "b"], fail_agent=1).stream("Q")))[-1]
    assert fin["type"] == "swarm_finished" and fin["agent_status"] == ["done", "failed"]


async def test_time_cut_keeps_partial_hits():
    fin = (await collect(make_swarm(subs=["a", "b"], stall_agent=1, max_seconds=0.5).stream("Q")))[-1]
    assert fin["agent_status"][1] == "cut" and fin["hits"]
    assert any(1 in h["agents"] for h in fin["hits"])   # the cut agent's snapshot is merged


async def test_closing_the_stream_cancels_the_agents():
    spy = SpyHarness(swarm_harness(), stall_agent=1)
    events = make_swarm(subs=["a", "b"], harness=spy).stream("Q")
    async for ev in events:
        if ev["type"] == "agent_event" and ev["agent"] == 1 and ev["event"]["type"] == "results_updated":
            break
    await events.aclose()
    assert spy.stall_closed


async def test_error_before_split_is_swarm_failed():
    class Boom:
        id = "boom"

        @property
        def budget(self):
            raise RuntimeError("no budget")

    evs = await collect(Swarm(Boom(), FakeSplitClient(["a", "b"]), SwarmConfig(), {}).stream("Q"))
    assert [e["type"] for e in evs] == ["swarm_failed"]
    assert evs[0]["error_type"] == "RuntimeError" and evs[0]["message"] == "no budget"


async def test_search_returns_a_search_result_for_the_eval_runner():
    res = await make_swarm(subs=["a", "b"]).search("Q", top_k=10, mode="swarm")
    assert res.mode == "swarm" and res.hits and res.usage.turns <= 2


async def test_search_raises_on_swarm_failed():
    class Boom:
        @property
        def budget(self):
            raise RuntimeError("no budget")

    with pytest.raises(RuntimeError, match="no budget"):
        await Swarm(Boom(), FakeSplitClient(["a", "b"]), SwarmConfig(), {}).search("Q")


def test_config_has_the_swarm_block_and_120s():
    cfg = load_config()
    assert cfg["budget"]["max_seconds"] == 120 and cfg["swarm"] == {"max_agents": 4, "agent_max_turns": 2, "rrf_k": 60}


def test_build_demo_swarm_reads_the_swarm_block():
    harness = swarm_harness()
    swarm = build_demo_swarm(load_config(), harness, {"wiki": "d"}, client=FakeSplitClient([]))
    assert swarm.cfg == SwarmConfig(max_agents=4, agent_max_turns=2, rrf_k=60)
    assert swarm.harness is harness
