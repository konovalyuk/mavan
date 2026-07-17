from app.rag.retrievers.file import FileRetriever
from app.rag.retrievers.vector import VectorRetriever
from app.rag.types import RetrievedChunk


def reciprocal_rank_fusion(result_lists: list[list[RetrievedChunk]], *, k: int = 60, top_k: int = 5) -> list[RetrievedChunk]:
    scores: dict[tuple[str, int], float] = {}
    docs: dict[tuple[str, int], RetrievedChunk] = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = doc.rrf_key
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            docs[key] = doc

    ranked_keys = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [
        RetrievedChunk(
            text=docs[key].text,
            score=scores[key],
            source=docs[key].source,
            start_offset=docs[key].start_offset,
            chunk_id=docs[key].chunk_id,
        )
        for key in ranked_keys
    ]


class HybridRetriever:
    def __init__(self, keyword: FileRetriever, vector: VectorRetriever, *, rrf_k: int = 60):
        self._keyword = keyword
        self._vector = vector
        self._rrf_k = rrf_k

    async def retrieve(self, question: str, *, top_k: int = 5, source_prefixes: tuple[str, ...] | None = None) -> list[RetrievedChunk]:
        keyword_hits = await self._keyword.retrieve(question, top_k=top_k, source_prefixes=source_prefixes)
        vector_hits = await self._vector.retrieve(question, top_k=top_k, source_prefixes=source_prefixes)
        return reciprocal_rank_fusion([keyword_hits, vector_hits], k=self._rrf_k, top_k=top_k)