import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.rag.types import RetrievedChunk


@dataclass
class VectorStore:
    chunks: list[RetrievedChunk]
    vectors: np.ndarray  # shape (n, dim), float32

    def search(self, query_vector: list[float] | np.ndarray, *, top_k: int = 5) -> list[RetrievedChunk]:
        if len(self.chunks) == 0:
            return []

        q = np.asarray(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        v_norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        v_norms = np.where(v_norms == 0, 1.0, v_norms)
        normalized = self.vectors / v_norms
        scores = normalized @ q

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(
                text=self.chunks[i].text,
                score=float(scores[i]),
                source=self.chunks[i].source,
                start_offset=self.chunks[i].start_offset,
                chunk_id=self.chunks[i].chunk_id,
            )
            for i in top_indices
        ]

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "text": c.text,
                "source": c.source,
                "start_offset": c.start_offset,
                "chunk_id": c.chunk_id,
            }
            for c in self.chunks
        ]
        (index_dir / "chunks.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        np.save(index_dir / "vectors.npy", self.vectors)

    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore":
        index_dir = index_dir.resolve()
        chunks_path = index_dir / "chunks.json"
        vectors_path = index_dir / "vectors.npy"
        if not chunks_path.is_file() or not vectors_path.is_file():
            raise FileNotFoundError(f"Vector index not found in {index_dir}")

        data = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = [
            RetrievedChunk(
                text=item["text"],
                score=0.0,
                source=item.get("source"),
                start_offset=item.get("start_offset"),
                chunk_id=item.get("chunk_id"),
            )
            for item in data
        ]
        vectors = np.load(vectors_path)
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks ({len(chunks)}) != vectors ({len(vectors)})")
        return cls(chunks=chunks, vectors=np.asarray(vectors, dtype=np.float32))
