from collections.abc import AsyncIterator
from openai import AsyncOpenAI
from app.llm.chat.schemas import ChatCompletionRequest, ChatCompletionResponse


class OpenAICompatibleChatAdapter:
    def __init__(self, *, api_key: str, base_url: str | None):
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            **request.to_provider_payload(stream=True),
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        response = await self._client.chat.completions.create(
            **request.to_provider_payload(stream=False),
        )
        text = (response.choices[0].message.content or "").strip()
        return ChatCompletionResponse(text=text, model=request.model)
