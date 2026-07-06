import asyncio
import json
import random
from collections.abc import AsyncIterator

from app.llm.chat.schemas import ChatCompletionRequest, ChatCompletionResponse


class MockChatAdapter:
    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        for word in self._reply(request).split():
            yield word + " "
            await asyncio.sleep(0.08)

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        # Simulate one tool call when tools are present and no tool result yet
        has_tool_result = any(m.role == "tool" for m in request.messages)
        if request.tools and not has_tool_result:
            user = next((m.content for m in request.messages if m.role == "user"), "")
            return ChatCompletionResponse(
                text=None,
                model=request.model,
                tool_calls=[{
                    "id": "call_mock_1",
                    "type": "function",
                    "function": {
                        "name": "search_notes",
                        "arguments": json.dumps({"query": user or "test", "top_k": 5}),
                    },
                }],
                finish_reason="tool_calls",
            )

        text = self._reply(request)
        return ChatCompletionResponse(text=text, model=request.model)

    def _reply(self, request: ChatCompletionRequest) -> str:
        user = next((m.content for m in request.messages if m.role == "user"), "")
        words = (user or "Hello from mock LLM").split()
        if not words:
            return "Mock response works."
        return " ".join(random.sample(words, min(len(words), 5)))
