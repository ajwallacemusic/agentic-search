from types import SimpleNamespace

import pytest

from agentic_search.core.types import ImagePart, TextPart
from agentic_search.models.anthropic import AnthropicClient
from agentic_search.models.base import ToolCall, ToolSpec
from agentic_search.models.llm import ChatMessage, ChatResponse, usage_with_cost
from agentic_search.testing import FakeLLMClient


class FakeMessages:
    def __init__(self, response):
        self.response, self.kwargs = response, None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def sdk(blocks, stop="tool_use", tin=1000, tout=200):
    resp = SimpleNamespace(content=blocks, stop_reason=stop,
                           usage=SimpleNamespace(input_tokens=tin, output_tokens=tout))
    return SimpleNamespace(messages=FakeMessages(resp))


TOOL = ToolSpec(name="lexical_search", description="d", parameters={"type": "object"})


def test_usage_with_cost():
    u = usage_with_cost(1_000_000, 500_000, (3.0, 15.0))
    assert u.cost_usd == pytest.approx(10.5)
    assert usage_with_cost(5, 5, None).cost_usd == 0.0


async def test_parses_text_and_tool_use():
    fake = sdk([SimpleNamespace(type="text", text="thinking"),
                SimpleNamespace(type="tool_use", id="tu1", name="lexical_search",
                                input={"source": "s", "text": "x"})])
    c = AnthropicClient("claude-sonnet-5-5", client=fake, price_per_mtok=(3.0, 15.0))
    assert c.id == "anthropic:claude-sonnet-5-5"
    r = await c.chat("sys", [ChatMessage(role="user", content="hi")], tools=[TOOL],
                     tool_choice="lexical_search")
    assert r.text == "thinking"
    assert r.tool_calls == [ToolCall(id="tu1", name="lexical_search",
                                     arguments={"source": "s", "text": "x"})]
    assert r.usage.input_tokens == 1000 and r.usage.cost_usd == pytest.approx(0.006)
    kw = fake.messages.kwargs
    assert kw["system"] == "sys" and kw["model"] == "claude-sonnet-5-5"
    assert kw["tools"] == [{"name": "lexical_search", "description": "d",
                            "input_schema": {"type": "object"}}]
    assert kw["tool_choice"] == {"type": "tool", "name": "lexical_search"}


async def test_message_conversion_merges_tool_results():
    fake = sdk([SimpleNamespace(type="text", text="ok")], stop="end_turn")
    c = AnthropicClient("m", client=fake)
    calls = [ToolCall(id="a", name="t", arguments={}), ToolCall(id="b", name="t", arguments={})]
    await c.chat("s", [
        ChatMessage(role="user", content="q"),
        ChatMessage(role="assistant", content="", tool_calls=calls),
        ChatMessage(role="tool", tool_call_id="a", content="r1"),
        ChatMessage(role="tool", tool_call_id="b", content="r2"),
    ])
    msgs = fake.messages.kwargs["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"] == [
        {"type": "tool_use", "id": "a", "name": "t", "input": {}},
        {"type": "tool_use", "id": "b", "name": "t", "input": {}},
    ]
    assert [b["tool_use_id"] for b in msgs[2]["content"]] == ["a", "b"]


async def test_images():
    fake = sdk([SimpleNamespace(type="text", text="")], stop="end_turn")
    content = [TextPart(text="look"), ImagePart(data=b"\x89PNG", mime="image/png")]
    await AnthropicClient("m", client=fake).chat("s", [ChatMessage(role="user", content=content)])
    blocks = fake.messages.kwargs["messages"][0]["content"]
    assert blocks[1]["type"] == "image" and blocks[1]["source"]["type"] == "base64"
    await AnthropicClient("m", client=fake, supports_images=False).chat(
        "s", [ChatMessage(role="user", content=content)])
    assert fake.messages.kwargs["messages"][0]["content"][1] == {"type": "text", "text": "[image omitted]"}


async def test_fake_llm_client():
    f = FakeLLMClient([ChatResponse(text="a")])
    assert (await f.chat("s", [ChatMessage(role="user", content="x")])).text == "a"
    assert f.requests[0]["system"] == "s"
    with pytest.raises(AssertionError):
        await f.chat("s", [])
