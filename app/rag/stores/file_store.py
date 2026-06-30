import json
from pathlib import Path

from app.rag.chunking import split_by_paragraphs
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


def load_and_chunk(data_dir: Path, *, max_chars: int = 2000) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    for rel_path, text in load_text_files(data_dir):
        for i, piece in enumerate(split_by_paragraphs(text, max_chars=max_chars)):
            source = f"file:{rel_path}" if i == 0 else f"file:{rel_path}#{i}"
            chunks.append(RetrievedChunk(text=piece, score=0.0, source=source))
    return chunks


def save_chunk_index(path: Path, chunks: list[RetrievedChunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"text": c.text, "score": c.score, "source": c.source} for c in chunks]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunk_index(path: Path) -> list[RetrievedChunk]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        RetrievedChunk(text=item["text"], score=float(item["score"]), source=item.get("source"))
        for item in data
    ]


def score_by_keyword_overlap(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    q_tokens = set(query.lower().split())
    scored: list[RetrievedChunk] = []
    for chunk in chunks:
        c_tokens = set(chunk.text.lower().split())
        overlap = len(q_tokens & c_tokens)
        score = overlap / max(len(q_tokens), 1)
        scored.append(RetrievedChunk(text=chunk.text, score=score, source=chunk.source))
    return scored
