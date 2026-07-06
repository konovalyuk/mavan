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
        msg = response.choices[0].message
        usage = None
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        tool_calls = None
        if msg.tool_calls:
            tool_calls = [tc.model_dump() for tc in msg.tool_calls]

        return ChatCompletionResponse(
            text=(msg.content or "").strip() or None,
            model=request.model,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=response.choices[0].finish_reason,
        )
