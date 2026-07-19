import asyncio
import json
import random
from collections.abc import AsyncIterator

from app.llm.chat.schemas import ChatCompletionRequest, ChatCompletionResponse


def _mock_args_for_tool(fn: dict, user: str) -> dict:
    params = fn.get("parameters") or {}
    required = params.get("required") or []
    props = params.get("properties") or {}
    args: dict = {}
    for key in required:
        spec = props.get(key) or {}
        if key in ("task", "query", "code"):
            args[key] = (user or "proceed")[:800]
        elif key == "url":
            args[key] = "https://example.com"
        elif spec.get("type") == "integer":
            args[key] = int(spec.get("default") or 1)
        elif spec.get("type") == "number":
            args[key] = float(spec.get("default") or 0)
        else:
            args[key] = "x"
    return args


class MockChatAdapter:
    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        for word in self._reply(request).split():
            yield word + " "
            await asyncio.sleep(0.08)

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Walk through declared tools one round each, then return text.

        Enough for local multi-agent smoke tests without inventing production fallbacks.
        """
        tools = request.tools or []
        n_tool_msgs = sum(1 for m in request.messages if m.role == "tool")
        if tools and n_tool_msgs < len(tools):
            fn = tools[n_tool_msgs]["function"]
            user = next((m.content for m in request.messages if m.role == "user"), "") or ""
            return ChatCompletionResponse(
                text=None,
                model=request.model,
                tool_calls=[{
                    "id": f"call_mock_{n_tool_msgs + 1}",
                    "type": "function",
                    "function": {
                        "name": fn["name"],
                        "arguments": json.dumps(_mock_args_for_tool(fn, user)),
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
