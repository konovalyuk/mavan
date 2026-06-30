from typing import Protocol


class EmbedProvider(Protocol):
    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]: ...