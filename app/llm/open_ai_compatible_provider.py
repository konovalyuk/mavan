from openai import AsyncOpenAI
from config import llm_settings


class OpenAICompatibleProvider:
    def __init__(self, base_url: str | None, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def stream(self, *, messages, model, max_tokens, temperature):
        stream = await self.client.chat.completions.create(
            model=model or llm_settings.MODEL,
            messages=[m.model_dump(exclude_none=True) for m in messages],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
