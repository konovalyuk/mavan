from app.rag.stores.file_store import score_bm25
from app.rag.types import RetrievedChunk


class MockRerankAdapter:
    """Локальный rerank: пересортировка по keyword overlap."""

    async def rerank(
            self,
            query: str,
            chunks: list[RetrievedChunk],
            *,
            model: str,
            top_k: int,
    ) -> list[RetrievedChunk]:
        _ = model
        if not chunks:
            return []
        scored = score_bm25(query, chunks)
        ranked = sorted(scored, key=lambda c: c.score, reverse=True)
        return ranked[:top_k]
