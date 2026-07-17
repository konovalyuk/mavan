from typing import Protocol
from app.rag.types import RetrievedChunk


class Retriever(Protocol):
    async def retrieve(self, question: str, *, top_k: int = 5, source_prefixes: tuple[str, ...] | None = None) -> list[RetrievedChunk]: ...
