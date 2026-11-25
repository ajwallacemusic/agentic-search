"""SID-2 Swarm driver against a scripted fake of SID's WebSocket (offline), inside a real harness."""

import asyncio
import contextlib
import json
import re

import pytest

pytest.importorskip("sid")

from agentic_search import Budget, Harness  # noqa: E402
from agentic_search.backends.files import FilesBackend  # noqa: E402
from agentic_search.core.harness import HarnessSettings  # noqa: E402
from agentic_search.core.types import Document, TextPart  # noqa: E402
from agentic_search.embedders.local import HashEmbedder  # noqa: E402
from demo.ignite.sid_swarm import SidSwarmDriver, rrf  # noqa: E402

DOCS = [("d1", "Aspirin relieves headache and mild pain."),
        ("d2", "Fever is a raised body temperature, often from infection."),
        ("d3", "Ibuprofen treats fever, pain and inflammation."),
        ("d4", "Castles were built in medieval Europe.")]


def ids_in(text):
    return re.findall(r'<doc id="([^"]+)"', text)


class FakeSid:
    """Plays server events in order. A step is an event dict, a callable(outputs) -> event, or
    ("wait", call_id...) to block until those calls were answered."""

    def __init__(self, steps):
        self.steps, self.sent = list(steps), []
        self.outputs, self.ended = {}, set()
        self._answered = asyncio.Event()

    async def send(self, message):
        self.sent.append(message)
        if message.get("type") == "response.inject":
            for item in message["input"]:
                self.outputs[item["call_id"]] = item.get("output", "")
                if item.get("end_agent"):
                    self.ended.add(item["call_id"])
            self._answered.set()

    async def recv_bytes(self):
        while self.steps:
            step = self.steps.pop(0)
            if isinstance(step, tuple) and step[0] == "wait":
                while not all(c in self.outputs for c in step[1:]):
                    self._answered.clear()
                    await self._answered.wait()
                continue
            event = step(self.outputs) if callable(step) else step
            return json.dumps(event).encode()
        await asyncio.Event().wait()  # nothing left: block like an idle socket


def call(agent, call_id, name, **args):
    return {"type": "response.output_item.done", "agent": {"agent_name": agent},
            "item": {"type": "function_call", "call_id": call_id, "name": name,
                     "arguments": json.dumps(args)}}


def report(agent, call_id, ids):
    return call(agent, call_id, "report_helpful_ids", ids=ids)


def fork(parent, children):
    return {"type": "response.output_item.done", "agent": {"agent_name": parent},
            "item": {"type": "multi_agent_call_output", "agent": {"agent_name": parent},
                     "output": [{"type": "text", "text": json.dumps({"agents": children})}]}}


CREATED = {"type": "response.created", "response": {"id": "resp_1"}}


def harness(fake, **settings):
    backend = FilesBackend.from_documents(
        "docs", [Document(doc_id=i, content=[TextPart(text=t)]) for i, t in DOCS],
        embedder=HashEmbedder())

    @contextlib.asynccontextmanager
    async def connect():
        yield fake

    driver = SidSwarmDriver(key="k", connect=connect, poll_s=0.05, max_agents=4, grace_s=0.3)
    return Harness([backend], driver, embedders=[backend.embedder],
                   settings=HarnessSettings(judge_delegate=False, **settings)), driver


async def test_swarm_answers_tools_forks_caches_and_fuses_reports():
    fake = FakeSid([
        CREATED,
        call("/root", "c1", "search", query="aspirin headache pain", limit=3),
        ("wait", "c1"),
        fork("/root", ["/root/a"]),
        call("/root/a", "c2", "text_search", query="fever", limit=3),
        ("wait", "c2"),
        lambda out: call("/root/a", "c3", "read", id=ids_in(out["c2"])[0]),
        ("wait", "c3"),
        lambda out: report("/root", "c4", ids_in(out["c1"])[:2]),
        lambda out: report("/root/a", "c5", ids_in(out["c2"])[:2] + ["not-an-id"]),
        ("wait", "c4", "c5"),
        {"type": "response.completed", "response": {"id": "resp_1", "agents": [], "usage": {
            "input_tokens": 1000, "output_tokens": 100,
            "input_tokens_details": {"cached_tokens": 400}}}},
    ])
    h, driver = harness(fake)
    res = await h.search("What treats a headache or a fever?", mode="model")

    create = fake.sent[0]
    assert create["type"] == "response.create" and create["model"] == "SID-2-Swarm"
    assert create["multi_agent"]["enabled"] and create["multi_agent"]["max_agents"] == 4
    assert [t["name"] for t in create["tools"]] == ["search", "text_search", "read",
                                                    "report_helpful_ids"]
    assert fake.ended == {"c4", "c5"}  # a valid report ends its agent
    assert "Fever" in fake.outputs["c3"] or "fever" in fake.outputs["c3"]
    assert all(m["response_id"] == "resp_1" for m in fake.sent[1:])

    # Reports are hit keys; the fused order is SID's RRF over the two reports.
    r_root = [k for k in res.keys() if k in {"docs:d1", "docs:d3"}]
    assert r_root and res.keys()[0] in {"docs:d1", "docs:d2", "docs:d3"}
    assert "docs:d4" not in res.keys()[:1]
    assert res.usage.tool_calls == 2 and res.usage.turns == 0  # read is served from the cache
    # cost: 600 fresh * 0.50 + 400 cached * 0.05 + 100 out * 2.00, per 1M
    assert res.usage.cost_usd == pytest.approx((600 * 0.50 + 400 * 0.05 + 100 * 2.00) / 1e6)
    assert not res.trace.of_type("judge")


async def test_a_bad_read_goes_back_to_the_model():
    fake = FakeSid([CREATED, call("/root", "c1", "read", id="999999"), ("wait", "c1"),
                    report("/root", "c2", []), ("wait", "c2"),
                    {"type": "response.completed", "response": {"id": "resp_1"}}])
    h, driver = harness(fake)
    res = await h.search("q", mode="model")
    assert "Invalid document id" in fake.outputs["c1"]
    assert res.trace.of_type("delegate")[0].data["note"].endswith("cost reads 0")


async def test_budget_refuses_searches_takes_reports_then_stops():
    fake = FakeSid([CREATED,
                    call("/root", "c1", "text_search", query="aspirin"), ("wait", "c1"),
                    fork("/root", ["/root/a"]),
                    call("/root/a", "c2", "text_search", query="fever"), ("wait", "c2"),
                    lambda out: report("/root", "c3", ids_in(out["c1"])), ("wait", "c3")])
    h, driver = harness(fake)
    res = await h.search("q", mode="model", budget=Budget(max_tool_calls=1))
    assert fake.outputs["c2"].startswith("[budget]") and "report_helpful_ids" in fake.outputs["c2"]
    assert res.keys()[0] == "docs:d1"  # the report sent after the budget ran out still counts
    assert "stopped early: budget" in res.trace.of_type("delegate")[0].data["note"]


async def test_server_failure_raises():
    fake = FakeSid([CREATED, {"type": "error", "error": {"code": "model_not_found"}}])
    h, driver = harness(fake)
    with pytest.raises(Exception, match="model_not_found"):
        await h.search("q", mode="model")


def test_rrf_matches_sid():
    assert rrf([["a", "b"], ["b", "c"], []]) == ["b", "a", "c"]


async def test_demo_build_runs_sid_swarm_in_model_mode_with_its_own_budget(monkeypatch):
    from agentic_search.embedders.local import HashEmbedder
    from demo.ignite.build import build_sid_harness_for, load_config
    from demo.ignite.tests.helpers import demo_data

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("IGNITE_GCP_PROJECT", "demo-project-1")
    data = demo_data()
    fake = FakeSid([CREATED, call("/root", "c1", "text_search", query="text 0"), ("wait", "c1"),
                    lambda out: report("/root", "c2", ids_in(out["c1"])[:3]), ("wait", "c2"),
                    {"type": "response.completed", "response": {"id": "resp_1"}}])

    @contextlib.asynccontextmanager
    async def connect():
        yield fake

    cfg = load_config()
    async with build_sid_harness_for(cfg, data.documents, data.descriptions, connect=connect,
                                     embedder=HashEmbedder()) as h:
        assert h.budget.max_tool_calls == cfg["sid_swarm"]["budget"]["max_tool_calls"]
        assert h.settings.judge_delegate is False and h.driver.id == "sid:SID-2-Swarm"
        res = await h.search(data.questions["hero"].text, mode="model")
    assert len(res.hits) == 3 and all(k.startswith("wiki:") for k in res.keys())
    assert fake.sent[0]["multi_agent"]["max_agents"] == cfg["sid_swarm"]["max_agents"]


def test_sid_swarm_is_an_eval_mode_that_runs_as_model_mode():
    from demo.ignite.eval import RUN_MODE

    assert RUN_MODE["sid-swarm"] == "model"


async def test_without_usage_the_cost_is_estimated_per_model_action():
    fake = FakeSid([CREATED, call("/root", "c1", "text_search", query="aspirin"), ("wait", "c1"),
                    {"type": "response.output_item.done", "agent": {"agent_name": "/root"},
                     "item": {"type": "multi_agent_call", "action": "spawn_agent"}},
                    report("/root", "c2", []), ("wait", "c2"),
                    {"type": "response.completed", "response": {"id": "resp_1"}}])
    h, driver = harness(fake)
    driver.est_cost_per_action = 0.001
    res = await h.search("q", mode="model")
    assert res.usage.cost_usd == pytest.approx(0.003)  # search + spawn + report
    assert "cost estimated: 3 model actions" in res.trace.of_type("delegate")[0].data["note"]
