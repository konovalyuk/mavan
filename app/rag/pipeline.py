from dataclasses import dataclass
from pathlib import Path

from config import rag_settings
from app.rag.context_budget import prepare_context_chunks
from app.rag.retriever import Retriever
from app.rag.retrievers.file import FileRetriever
from app.rag.retrievers.vector import VectorRetriever
from app.rag.retrievers.hybrid import HybridRetriever
from app.rag.types import RetrievedChunk
from app.rag.rerank_step import maybe_rerank

DEFAULT_TOP_K = 5

_pipeline: "RagPipeline | None" = None


def build_retriever() -> Retriever:
    kind = rag_settings.RETRIEVER.lower()
    chunks_path = rag_settings.chunks_path
    if kind == "file":
        return FileRetriever(index_path=chunks_path)
    if kind == "vector":
        return VectorRetriever(Path(rag_settings.VECTOR_INDEX_PATH))
    if kind == "hybrid":
        keyword = FileRetriever(index_path=chunks_path)
        vector = VectorRetriever(Path(rag_settings.VECTOR_INDEX_PATH))
        return HybridRetriever(keyword, vector, rrf_k=rag_settings.RRF_K)
    raise ValueError(f"Unknown RAG_RETRIEVER: {kind!r}. Use file|vector|hybrid.")


@dataclass
class RagPipeline:
    retriever: Retriever

    async def retrieve_chunks(self, question: str, *, top_k: int = DEFAULT_TOP_K, source_prefixes: tuple[str, ...] | None = None) -> list[RetrievedChunk]:
        fetch_k = rag_settings.FETCH_K if rag_settings.RERANK_ENABLED else top_k
        raw = await self.retriever.retrieve(question, top_k=fetch_k, source_prefixes=source_prefixes)
        chunks = await maybe_rerank(question, raw, top_k=top_k)
        return prepare_context_chunks(chunks, min_score=rag_settings.MIN_SCORE, max_chars=rag_settings.CONTEXT_MAX_CHARS, min_relative=rag_settings.MIN_RELATIVE_SCORE)


def get_default_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline(retriever=build_retriever())
    return _pipeline


def reset_pipeline() -> None:
    global _pipeline
    _pipeline = None
