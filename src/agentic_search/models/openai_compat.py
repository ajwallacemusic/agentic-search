"""OpenAI-compatible chat completions adapter: OpenAI, vLLM, Ollama, Together, Baseten
(e.g. SID-1), and most open-model servers. Needs the `openai` extra unless a client is injected."""

from __future__ import annotations

import base64
import json
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentic_search.embedders.auth import Auth

from agentic_search.core.types import Content, ImagePart, image_bytes, text_of
from agentic_search.models.base import ToolCall, ToolSpec
from agentic_search.models.llm import ChatMessage, ChatResponse, usage_with_cost


def _as_text(content: str | list[Content]) -> str:
    return content if isinstance(content, str) else text_of(content)


def _parse_args(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"_invalid_json": raw}
    return value if isinstance(value, dict) else {"_invalid_json": raw}


class OpenAICompatClient:
    def __init__(self, model: str, *, base_url: str | None = None, api_key: str | None = None,
                 client: Any = None, supports_images: bool = False,
                 price_per_mtok: tuple[float, float] | None = None, id: str | None = None,
                 max_tokens_param: str = "max_tokens", max_retries: int = 3,
                 auth: Auth | None = None):
        if client is None:
            from openai import AsyncOpenAI
            key = api_key or os.environ.get("OPENAI_API_KEY") or "unused"
            client = AsyncOpenAI(base_url=base_url, api_key=key, max_retries=max_retries)
        self.model = model
        self._client = client
        self.supports_images = supports_images
        self.price = price_per_mtok
        self.max_tokens_param = max_tokens_param
        self.auth = auth
        self.id = id or f"openai:{model}"

    async def chat(self, system: str, messages: list[ChatMessage], *,
                   tools: list[ToolSpec] | None = None, tool_choice: str | None = None,
                   max_tokens: int = 4096) -> ChatResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *self._messages(messages)],
            self.max_tokens_param: max_tokens,
        }
        if tools:
            kwargs["tools"] = [{"type": "function", "function": {
                "name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools]
        if tool_choice:
            kwargs["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}
        if self.auth is not None:
            # Headers per call, so a short-lived token is refreshed rather than frozen at startup.
            kwargs["extra_headers"] = await self.auth.headers()
        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
        calls = [ToolCall(id=tc.id, name=tc.function.name, arguments=_parse_args(tc.function.arguments))
                 for tc in (msg.tool_calls or [])]
        u = resp.usage
        usage = usage_with_cost(getattr(u, "prompt_tokens", 0) or 0,
                                getattr(u, "completion_tokens", 0) or 0, self.price)
        return ChatResponse(text=msg.content or "", tool_calls=calls, usage=usage,
                            stop_reason=choice.finish_reason)

    def _messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "tool":
                out.append({"role": "tool", "tool_call_id": m.tool_call_id,
                            "content": _as_text(m.content)})
            elif m.role == "assistant":
                entry: dict[str, Any] = {"role": "assistant", "content": _as_text(m.content) or None}
                if m.tool_calls:
                    entry["tool_calls"] = [{"id": c.id, "type": "function", "function": {
                        "name": c.name, "arguments": json.dumps(c.arguments)}} for c in m.tool_calls]
                out.append(entry)
            else:
                out.append({"role": "user", "content": self._user_content(m.content)})
        return out

    def _user_content(self, content: str | list[Content]) -> str | list[dict[str, Any]]:
        if isinstance(content, str):
            return content
        parts: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, ImagePart):
                if not self.supports_images:
                    parts.append({"type": "text", "text": "[image omitted]"})
                    continue
                if part.uri is not None and part.uri.startswith(("http://", "https://")):
                    url = part.uri
                else:
                    url = f"data:{part.mime};base64,{base64.b64encode(image_bytes(part)).decode()}"
                parts.append({"type": "image_url", "image_url": {"url": url}})
            else:
                parts.append({"type": "text", "text": text_of([part])})
        return parts
