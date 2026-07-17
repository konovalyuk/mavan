from pathlib import Path

from app.rag.sources import matches_source_prefix
from app.rag.stores.file_store import load_chunk_index, score_bm25
from app.rag.types import RetrievedChunk


class FileRetriever:
    """Keyword search over pre-chunked text files."""

    def __init__(self, *, index_path: Path):
        self._chunks = load_chunk_index(index_path) if index_path.is_file() else []

    async def retrieve(self, question: str, *, top_k: int = 5, source_prefixes: tuple[str, ...] | None = None) -> list[RetrievedChunk]:
        if not self._chunks:
            return []

        chunks = self._chunks
        if source_prefixes:
            chunks = [c for c in chunks if matches_source_prefix(c.source, source_prefixes)]
            if not chunks:
                return []

        ranked = sorted(score_bm25(question, chunks), key=lambda c: c.score, reverse=True)
        return [c for c in ranked if c.score > 0][:top_k] or ranked[:top_k]