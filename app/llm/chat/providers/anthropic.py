from collections.abc import AsyncIterator
from anthropic import AsyncAnthropic
from app.llm.chat.schemas import ChatCompletionRequest, ChatCompletionResponse


class AnthropicChatAdapter:
    def __init__(self, *, api_key: str, base_url: str | None):
        self._client = AsyncAnthropic(api_key=api_key, base_url=base_url)

    def _split_messages(self, request: ChatCompletionRequest):
        system = "\n".join(
            m.content for m in request.messages if m.role == "system" and m.content
        )
        chat_messages = [
            {"role": m.role, "content": m.content}
            for m in request.messages
            if m.role in ("user", "assistant") and m.content is not None
        ]
        return system or None, chat_messages

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        system, chat_messages = self._split_messages(request)
        response = await self._client.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system=system,
            messages=chat_messages,
        )
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        return ChatCompletionResponse(text=text, model=request.model)

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        system, chat_messages = self._split_messages(request)
        async with self._client.messages.stream(
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=system or None,
                messages=chat_messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text
