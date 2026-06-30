from app.rag.types import RetrievedChunk


def filter_by_score(chunks: list[RetrievedChunk], *, min_score: float = 0.0) -> list[RetrievedChunk]:
    if min_score <= 0:
        return chunks
    kept = [c for c in chunks if c.score >= min_score]
    return kept if kept else chunks[:1]


def filter_by_relative_score(chunks: list[RetrievedChunk], *, min_relative: float = 0.8) -> list[RetrievedChunk]:
    if not chunks:
        return []
    best = max(c.score for c in chunks)
    if best <= 0:
        return chunks
    threshold = best * min_relative
    kept = [c for c in chunks if c.score >= threshold]
    return kept if kept else chunks[:1]


def trim_chunks(chunks: list[RetrievedChunk], *, max_chars: int = 8000) -> list[RetrievedChunk]:
    selected: list[RetrievedChunk] = []
    total = 0
    for c in sorted(chunks, key=lambda x: x.score, reverse=True):
        if total + len(c.text) > max_chars:
            break
        selected.append(c)
        total += len(c.text)
    return selected


def prepare_context_chunks(chunks: list[RetrievedChunk], *, min_score: float = 0.0, max_chars: int = 8000,
                           min_relative: float = 0.0) -> list[RetrievedChunk]:
    filtered = filter_by_score(chunks, min_score=min_score)
    if min_relative > 0:
        filtered = filter_by_relative_score(filtered, min_relative=min_relative)
    return trim_chunks(filtered, max_chars=max_chars)
