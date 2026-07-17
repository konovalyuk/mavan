from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.rag.sources import matches_source_prefix
from app.rag.stores.file_store import load_chunk_index, save_chunk_index
from app.rag.types import RetrievedChunk


@dataclass
class VectorStore:
    chunks: list[RetrievedChunk]
    vectors: np.ndarray  # shape (n, dim), float32

    def search(self, query_vector: list[float] | np.ndarray, *, top_k: int = 5, source_prefixes: tuple[str, ...] | None = None) -> list[RetrievedChunk]:
        if not self.chunks:
            return []

        rows = range(len(self.chunks))
        if source_prefixes:
            rows = [i for i in rows if matches_source_prefix(self.chunks[i].source, source_prefixes)]
            if not rows:
                return []

        q = np.asarray(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        mat = self.vectors[list(rows)]
        v_norms = np.linalg.norm(mat, axis=1, keepdims=True)
        v_norms = np.where(v_norms == 0, 1.0, v_norms)
        scores = (mat / v_norms) @ q

        k = min(top_k, len(rows))
        top_local = np.argsort(scores)[::-1][:k]
        return [
            RetrievedChunk(
                text=self.chunks[rows[j]].text,
                score=float(scores[j]),
                source=self.chunks[rows[j]].source,
                start_offset=self.chunks[rows[j]].start_offset,
                chunk_id=self.chunks[rows[j]].chunk_id,
            )
            for j in top_local
        ]

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        save_chunk_index(index_dir / "chunks.json", self.chunks)
        np.save(index_dir / "vectors.npy", self.vectors)

    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore":
        index_dir = index_dir.resolve()
        chunks_path = index_dir / "chunks.json"
        vectors_path = index_dir / "vectors.npy"
        if not chunks_path.is_file() or not vectors_path.is_file():
            raise FileNotFoundError(f"Vector index not found in {index_dir}")

        chunks = load_chunk_index(chunks_path)
        vectors = np.load(vectors_path)
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks ({len(chunks)}) != vectors ({len(vectors)})")
        return cls(chunks=chunks, vectors=np.asarray(vectors, dtype=np.float32))
