from openai import AsyncOpenAI


class OpenAICompatibleEmbedAdapter:
    def __init__(self, *, api_key: str, base_url: str | None):
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=model,
            input=texts,
        )
        items = response.data
        if len(items) != len(texts):
            raise ValueError(f"Expected {len(texts)} embeddings, got {len(items)}")

        if all(item.index is not None for item in items):
            by_index = {item.index: item.embedding for item in items}
            return [by_index[i] for i in range(len(texts))]

        return [item.embedding for item in items]
