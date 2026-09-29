"""Anthropic Messages API adapter. Needs the `anthropic` extra unless a client is injected."""

from __future__ import annotations

import base64
from typing import Any

from agentic_search.core.types import Content, ImagePart, image_bytes, text_of
from agentic_search.models.base import ToolCall, ToolSpec
from agentic_search.models.llm import ChatMessage, ChatResponse, usage_with_cost


def _as_text(content: str | list[Content]) -> str:
    return content if isinstance(content, str) else text_of(content)


def _image_block(part: ImagePart) -> dict[str, Any]:
    if part.uri is not None and part.uri.startswith(("http://", "https://")):
        return {"type": "image", "source": {"type": "url", "url": part.uri}}
    data = base64.b64encode(image_bytes(part)).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": part.mime, "data": data}}


class AnthropicClient:
    def __init__(self, model: str, *, client: Any = None, api_key: str | None = None,
                 supports_images: bool = True, price_per_mtok: tuple[float, float] | None = None,
                 id: str | None = None, max_retries: int = 3):
        if client is None:
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic(api_key=api_key, max_retries=max_retries)
        self.model = model
        self._client = client
        self.supports_images = supports_images
        self.price = price_per_mtok
        self.id = id or f"anthropic:{model}"

    async def chat(self, system: str, messages: list[ChatMessage], *,
                   tools: list[ToolSpec] | None = None, tool_choice: str | None = None,
                   max_tokens: int = 4096) -> ChatResponse:
        kwargs: dict[str, Any] = {"model": self.model, "max_tokens": max_tokens, "system": system,
                                  "messages": self._messages(messages)}
        if tools:
            kwargs["tools"] = [{"name": t.name, "description": t.description,
                                "input_schema": t.parameters} for t in tools]
        if tool_choice:
            kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}
        resp = await self._client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(id=b.id, name=b.name, arguments=dict(b.input or {}))
                 for b in resp.content if b.type == "tool_use"]
        return ChatResponse(text=text, tool_calls=calls, stop_reason=resp.stop_reason,
                            usage=usage_with_cost(resp.usage.input_tokens,
                                                  resp.usage.output_tokens, self.price))

    def _messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "tool":
                block = {"type": "tool_result", "tool_use_id": m.tool_call_id,
                         "content": _as_text(m.content)}
                prev = out[-1] if out else None
                if (prev and prev["role"] == "user" and isinstance(prev["content"], list)
                        and all(b.get("type") == "tool_result" for b in prev["content"])):
                    prev["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            elif m.role == "assistant":
                blocks: list[dict[str, Any]] = []
                if _as_text(m.content):
                    blocks.append({"type": "text", "text": _as_text(m.content)})
                blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                           for c in m.tool_calls]
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": "user", "content": self._user_content(m.content)})
        return out

    def _user_content(self, content: str | list[Content]) -> str | list[dict[str, Any]]:
        if isinstance(content, str):
            return content
        blocks: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, ImagePart):
                blocks.append(_image_block(part) if self.supports_images
                              else {"type": "text", "text": "[image omitted]"})
            else:
                blocks.append({"type": "text", "text": text_of([part])})
        return blocks
