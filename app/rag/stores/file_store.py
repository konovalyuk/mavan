import json
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


def load_and_chunk(data_dir: Path, *, max_chars: int = 2000, overlap_chars: int = 200, strategy: str = "paragraph",
                   step: int = 1000) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    for rel_path, text in load_text_files(data_dir):
        if strategy == "sliding":
            parts = split_sliding_window(text, size=max_chars, step=step)
        else:
            parts = [(0, p) for p in split_by_paragraphs(text, max_chars=max_chars, overlap_chars=overlap_chars)]
        for start, piece in parts:
            source = f"file:{rel_path}"
            chunk_id = f"{source}@{start}"
            chunks.append(
                RetrievedChunk(text=piece, score=0.0, source=source, start_offset=start, chunk_id=chunk_id)
            )
    return chunks


def save_chunk_index(path: Path, chunks: list[RetrievedChunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "text": c.text,
            "score": c.score,
            "source": c.source,
            "start_offset": c.start_offset,
            "chunk_id": c.chunk_id,
        }
        for c in chunks
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunk_index(path: Path) -> list[RetrievedChunk]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        RetrievedChunk(
            text=item["text"],
            score=float(item["score"]),
            source=item.get("source"),
            start_offset=item.get("start_offset"),
            chunk_id=item.get("chunk_id"),
        )
        for item in data
    ]


def score_by_keyword_overlap(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    q_tokens = set(query.lower().split())
    scored: list[RetrievedChunk] = []
    for chunk in chunks:
        c_tokens = set(chunk.text.lower().split())
        overlap = len(q_tokens & c_tokens)
        score = overlap / max(len(q_tokens), 1)
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
