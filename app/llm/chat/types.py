from dataclasses import dataclass
from typing import AsyncIterator, Protocol
from app.models.data_models import ChatMessage


@dataclass(frozen=True)
class ChatRequest:
    messages: list[ChatMessage]
    model: str
    max_tokens: int
    temperature: float
    tools: list[dict] | None = None


@dataclass(frozen=True)
class ChatResponse:
    text: str
    model: str | None = None
    reasoning: str | None = None


class ChatProvider(Protocol):
    async def stream(self, request: ChatRequest) -> AsyncIterator[str]: ...

    async def complete(self, request: ChatRequest) -> ChatResponse: ...
