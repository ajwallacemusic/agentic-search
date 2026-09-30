from types import SimpleNamespace

from agentic_search.core.types import ImagePart, TextPart
from agentic_search.models.base import ToolCall, ToolSpec
from agentic_search.models.llm import ChatMessage
from agentic_search.models.openai_compat import OpenAICompatClient


class FakeCompletions:
    def __init__(self, response):
        self.response, self.kwargs = response, None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def sdk(content="hi", tool_calls=None, usage=(10, 5)):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")],
                           usage=SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1]))
    return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(resp)))


def tc(id, name, args):
    return SimpleNamespace(id=id, function=SimpleNamespace(name=name, arguments=args))


async def test_parses_tool_calls_and_usage():
    fake = sdk(tool_calls=[tc("a", "lexical_search", '{"source":"s","text":"x"}'),
                           tc("b", "lexical_search", "{oops")])
    c = OpenAICompatClient("qwen3", client=fake, price_per_mtok=(1.0, 2.0))
    assert c.id == "openai:qwen3"
    r = await c.chat("sys", [ChatMessage(role="user", content="q")],
                     tools=[ToolSpec(name="lexical_search", description="d", parameters={})],
                     tool_choice="lexical_search", max_tokens=100)
    assert r.text == "hi" and r.tool_calls[0].arguments == {"source": "s", "text": "x"}
    assert r.tool_calls[1].arguments == {"_invalid_json": "{oops"}
    assert r.usage.input_tokens == 10 and r.usage.cost_usd == (10 * 1 + 5 * 2) / 1e6
    kw = fake.chat.completions.kwargs
    assert kw["messages"][0] == {"role": "system", "content": "sys"}
    assert kw["tools"][0]["function"]["name"] == "lexical_search"
    assert kw["tool_choice"] == {"type": "function", "function": {"name": "lexical_search"}}
    assert kw["max_tokens"] == 100


async def test_message_conversion():
    fake = sdk(content=None)
    c = OpenAICompatClient("m", client=fake, supports_images=True, max_tokens_param="max_completion_tokens")
    await c.chat("s", [
        ChatMessage(role="user", content=[TextPart(text="look"), ImagePart(data=b"x", mime="image/png")]),
        ChatMessage(role="assistant", content="", tool_calls=[ToolCall(id="a", name="t", arguments={"k": 1})]),
        ChatMessage(role="tool", tool_call_id="a", content="result"),
    ])
    kw = fake.chat.completions.kwargs
    user, asst, tool = kw["messages"][1:]
    assert user["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert asst == {"role": "assistant", "content": None, "tool_calls": [
        {"id": "a", "type": "function", "function": {"name": "t", "arguments": '{"k": 1}'}}]}
    assert tool == {"role": "tool", "tool_call_id": "a", "content": "result"}
    assert "max_completion_tokens" in kw


async def test_missing_usage_and_content():
    fake = sdk(content=None, tool_calls=None)
    fake.chat.completions.response.usage = None
    r = await OpenAICompatClient("m", client=fake).chat("s", [ChatMessage(role="user", content="q")])
    assert r.text == "" and r.tool_calls == [] and r.usage.input_tokens == 0


class CountingAuth:
    def __init__(self):
        self.calls = 0

    async def headers(self):
        self.calls += 1
        return {"Authorization": f"Bearer tok{self.calls}"}


async def test_auth_headers_are_fetched_on_every_call():
    fake = sdk()
    auth = CountingAuth()
    c = OpenAICompatClient("m", client=fake, auth=auth)
    await c.chat("s", [ChatMessage(role="user", content="q")])
    assert fake.chat.completions.kwargs["extra_headers"] == {"Authorization": "Bearer tok1"}
    await c.chat("s", [ChatMessage(role="user", content="q")])
    assert fake.chat.completions.kwargs["extra_headers"] == {"Authorization": "Bearer tok2"}


async def test_no_auth_sends_no_extra_headers():
    fake = sdk()
    c = OpenAICompatClient("m", client=fake)
    await c.chat("s", [ChatMessage(role="user", content="q")])
    assert "extra_headers" not in fake.chat.completions.kwargs
