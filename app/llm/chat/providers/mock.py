import asyncio
import random
from collections.abc import AsyncIterator

from app.llm.chat.types import ChatRequest, ChatResponse


class MockChatAdapter:
    async def stream(self, request: ChatRequest) -> AsyncIterator[str]:
        for word in self._reply(request).split():
            yield word + " "
            await asyncio.sleep(0.08)

    async def complete(self, request: ChatRequest) -> ChatResponse:
        text = self._reply(request)
        return ChatResponse(text=text, model=request.model)

    def _reply(self, request: ChatRequest) -> str:
        user = next((m.content for m in request.messages if m.role == "user"), "")
        words = (user or "Hello from mock LLM").split()
        if not words:
            return "Mock response works."
        return " ".join(random.sample(words, min(len(words), 5)))