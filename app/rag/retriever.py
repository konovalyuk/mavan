from typing import Protocol
from app.rag.types import RetrievedChunk


class Retriever(Protocol):
    async def retrieve(self, question: str, *, top_k: int = 5) -> list[RetrievedChunk]: ...
