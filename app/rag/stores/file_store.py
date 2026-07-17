import json
import math
from collections import Counter
from pathlib import Path

from app.rag.chunking import split_by_paragraphs, split_sliding_window
from app.rag.types import RetrievedChunk

TEXT_EXTENSIONS = {".txt", ".md"}


def load_text_files(data_dir: Path) -> list[tuple[str, str]]:
    data_dir = data_dir.resolve()
    out: list[tuple[str, str]] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        rel = path.relative_to(data_dir).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        out.append((rel, text))
    return out


def chunk_text(text: str, *, source: str, max_chars: int = 2000, overlap_chars: int = 200, strategy: str = "paragraph", step: int = 1000) -> list[RetrievedChunk]:
    if strategy == "sliding":
        return [
            RetrievedChunk(text=piece, score=0.0, source=source, start_offset=start, chunk_id=f"{source}@{start}")
            for start, piece in split_sliding_window(text, size=max_chars, step=step)
        ]
    return [
        RetrievedChunk(text=piece, score=0.0, source=source, start_offset=idx, chunk_id=f"{source}@{idx}")
        for idx, piece in enumerate(split_by_paragraphs(text, max_chars=max_chars, overlap_chars=overlap_chars))
    ]


def chunks_to_json(chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "text": c.text,
            "score": c.score,
            "source": c.source,
            "start_offset": c.start_offset,
            "chunk_id": c.chunk_id,
        }
        for c in chunks
    ]


def chunks_from_json(data: list) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            text=item["text"],
            score=float(item.get("score", 0.0)),
            source=item.get("source"),
            start_offset=item.get("start_offset"),
            chunk_id=item.get("chunk_id"),
        )
        for item in data
    ]


def save_chunk_index(path: Path, chunks: list[RetrievedChunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chunks_to_json(chunks), ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunk_index(path: Path) -> list[RetrievedChunk]:
    return chunks_from_json(json.loads(path.read_text(encoding="utf-8")))


def score_bm25(query: str, chunks: list[RetrievedChunk], *, k1: float = 1.5, b: float = 0.75) -> list[RetrievedChunk]:
    if not chunks:
        return []
    docs = [c.text.lower().split() for c in chunks]
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n
    q_terms = query.lower().split()
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(set(doc))
    scored: list[RetrievedChunk] = []
    for chunk, doc in zip(chunks, docs):
        dl = len(doc)
        tf = Counter(doc)
        score = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            idf = math.log((n - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
            freq = tf[term]
            score += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * dl / avgdl))
        scored.append(
            RetrievedChunk(
                text=chunk.text,
                score=score,
                source=chunk.source,
                start_offset=chunk.start_offset,
                chunk_id=chunk.chunk_id,
            )
        )
    return scored
