from typing import Protocol


class EmbedProvider(Protocol):
    async def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]: ...