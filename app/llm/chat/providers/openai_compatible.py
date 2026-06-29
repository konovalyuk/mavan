from collections.abc import AsyncIterator
from openai import AsyncOpenAI
from app.llm.chat.types import ChatRequest, ChatResponse


class OpenAICompatibleChatAdapter:
    def __init__(self, *, api_key: str, base_url: str | None, default_model: str):
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._default_model = default_model

    async def stream(self, request: ChatRequest) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=request.model,
            messages=[m.model_dump(exclude_none=True) for m in request.messages],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def complete(self, request: ChatRequest) -> ChatResponse:
        response = await self._client.chat.completions.create(
            model=request.model,
            messages=[m.model_dump(exclude_none=True) for m in request.messages],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=False,
        )
        text = (response.choices[0].message.content or "").strip()
        return ChatResponse(text=text, model=request.model)
