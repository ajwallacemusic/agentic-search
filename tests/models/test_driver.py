from agentic_search.core.types import Budget, ImagePart, Query, TextPart
from agentic_search.models.base import Action, Decision, Driver, PlannerView, ToolCall, ToolSpec
from agentic_search.models.driver import FINISH_TOOL, ToolCallingDriver, render_planner_view
from agentic_search.models.llm import ChatResponse
from agentic_search.testing import FakeLLMClient

TOOLS = [ToolSpec(name="lexical_search", description="d", parameters={})]


def tc(i, name="lexical_search", **args):
    return ToolCall(id=f"id{i}", name=name, arguments=args)


def view(question=None):
    return PlannerView(question=question or Query.of("what treats headache?"), turn=1,
                       manifest_summary="## source `docs`", digest="Turn 0: stuff",
                       directive=Decision(action=Action.REFINE, note="be precise"), max_calls=2)


def test_render_planner_view():
    text = render_planner_view(view())
    for s in ("what treats headache?", "## source `docs`", "# Turn 1", "Turn 0: stuff",
              "REFINE", "be precise", "up to 2 tool calls"):
        assert s in text


async def test_plan_truncates_and_passes_tools():
    client = FakeLLMClient([ChatResponse(text="plan", tool_calls=[tc(1), tc(2), tc(3)])])
    d = ToolCallingDriver(client, max_calls_per_turn=8)
    assert isinstance(d, Driver) and d.id == "fake-llm"
    res = await d.plan(view(), TOOLS)
    assert len(res.calls) == 2 and res.note == "plan"
    req = client.requests[0]
    assert req["tools"] == TOOLS and "up to 2" in req["system"]
    assert isinstance(req["messages"][0].content, str)


async def test_plan_includes_question_images_when_supported():
    q = Query(content=[TextPart(text="similar scans?"), ImagePart(data=b"x")])
    client = FakeLLMClient([ChatResponse()], supports_images=True)
    await ToolCallingDriver(client).plan(view(q), TOOLS)
    content = client.requests[0]["messages"][0].content
    assert isinstance(content, list) and isinstance(content[1], ImagePart)


class Runtime:
    def __init__(self):
        self.batches = []

    async def call(self, calls):
        self.batches.append(calls)
        return [f"result for {c.id}" for c in calls]


async def test_delegate_loop_until_finish():
    client = FakeLLMClient([
        ChatResponse(tool_calls=[tc(1, source="docs", text="headache")]),
        ChatResponse(text="done", tool_calls=[tc(2, name="finish", ranked_keys=["docs:d1"])]),
    ])
    rt = Runtime()
    res = await ToolCallingDriver(client).run_delegate(Query.of("q"), TOOLS, rt, Budget(),
                                                        context="## source `docs`")
    assert res.ranked_keys == ["docs:d1"] and res.note == "done"
    assert len(rt.batches) == 1
    second = client.requests[1]["messages"]
    assert [m.role for m in second] == ["user", "assistant", "tool"]
    assert second[2].content == "result for id1"
    assert "## source `docs`" in second[0].content
    assert client.requests[0]["tools"][-1] == FINISH_TOOL


async def test_delegate_forces_finish_after_budget_turns():
    client = FakeLLMClient([
        ChatResponse(tool_calls=[tc(1)]),
        ChatResponse(tool_calls=[tc(2)]),
        ChatResponse(tool_calls=[tc(3, name="finish", ranked_keys=["a", 7])]),
    ])
    res = await ToolCallingDriver(client).run_delegate(Query.of("q"), TOOLS, Runtime(),
                                                        Budget(max_turns=1))
    assert res.ranked_keys == ["a", "7"]
    assert client.requests[2]["tool_choice"] == "finish"
    assert client.requests[2]["tools"] == [FINISH_TOOL]


async def test_delegate_stops_when_model_stops_calling_tools():
    client = FakeLLMClient([ChatResponse(text="nothing to do")])
    res = await ToolCallingDriver(client).run_delegate(Query.of("q"), TOOLS, Runtime(), Budget())
    assert res.ranked_keys == [] and res.note == "nothing to do"
