from typing import Any, Literal
from pydantic import BaseModel, ConfigDict
# from collections.abc import AsyncIterator


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request for provider adapters."""

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage]
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None
    stream: bool | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None

    def to_provider_payload(self, *, stream: bool | None = None) -> dict[str, Any]:
        payload = self.model_dump(exclude_none=True)
        if stream is not None:
            payload["stream"] = stream
        else:
            payload.pop("stream", None)
        return payload


class ChatCompletionResponse(BaseModel):
    text: str
    model: str | None = None
    reasoning: str | None = None
