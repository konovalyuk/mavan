from pathlib import Path

from app.rag.stores.file_store import load_and_chunk, load_chunk_index, score_by_keyword_overlap
from app.rag.types import RetrievedChunk


class FileRetriever:
    """Keyword search over pre-chunked text files."""

    def __init__(self, data_dir: Path, *, index_path: Path | None = None):
        if index_path and index_path.is_file():
            self._chunks = load_chunk_index(index_path)
        else:
            self._chunks = load_and_chunk(data_dir)

    async def retrieve(self, question: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        if not self._chunks:
            return []
        scored = score_by_keyword_overlap(question, self._chunks)
        ranked = sorted(scored, key=lambda c: c.score, reverse=True)
        return [c for c in ranked if c.score > 0][:top_k] or ranked[:top_k]
