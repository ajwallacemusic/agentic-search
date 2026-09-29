"""Provider-neutral chat interface that drivers and LLM judges are built on."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agentic_search.core.types import Content, ModelUsage
from agentic_search.models.base import ToolCall, ToolSpec


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str | list[Content] = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None


class ChatResponse(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: ModelUsage = Field(default_factory=ModelUsage)
    stop_reason: str | None = None


@runtime_checkable
class LLMClient(Protocol):
    id: str
    supports_images: bool

    async def chat(self, system: str, messages: list[ChatMessage], *,
                   tools: list[ToolSpec] | None = None, tool_choice: str | None = None,
                   max_tokens: int = 4096) -> ChatResponse: ...


def usage_with_cost(input_tokens: int, output_tokens: int,
                    price_per_mtok: tuple[float, float] | None) -> ModelUsage:
    cost = 0.0
    if price_per_mtok is not None:
        cost = (input_tokens * price_per_mtok[0] + output_tokens * price_per_mtok[1]) / 1_000_000
    return ModelUsage(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost)
