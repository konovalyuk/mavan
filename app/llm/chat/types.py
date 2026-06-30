from typing import AsyncIterator, Protocol

from app.llm.chat.schemas import ChatCompletionRequest, ChatCompletionResponse


class ChatProvider(Protocol):
    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]: ...

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse: ...
