from typing import Protocol

from app.rag.types import RetrievedChunk


class RerankProvider(Protocol):
    async def rerank(
            self,
            query: str,
            chunks: list[RetrievedChunk],
            *,
            model: str,
            top_k: int,
    ) -> list[RetrievedChunk]: ...
