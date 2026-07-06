def split_by_paragraphs(text: str, *, max_chars: int = 2000, overlap_chars: int = 200) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        piece = f"{buf}\n\n{para}".strip() if buf else para
        if len(piece) <= max_chars:
            buf = piece
            continue
        if buf:
            chunks.append(buf)
        while len(para) > max_chars:
            chunks.append(para[:max_chars])
            para = para[max_chars - overlap_chars:]
        buf = para
    if buf:
        chunks.append(buf)
    return chunks


def split_sliding_window(text: str, *, size: int = 2000, step: int = 1000) -> list[tuple[int, str]]:
    """Returns (start_offset, chunk_text)."""
    if not text:
        return []
    if len(text) <= size:
        return [(0, text)]
    out: list[tuple[int, str]] = []
    start = 0
    while start < len(text):
        piece = text[start:start + size]
        if piece.strip():
            out.append((start, piece))
        if start + size >= len(text):
            break
        start += step
    return out
