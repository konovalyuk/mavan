from app.rag.types import RetrievedChunk
from app.services.rag_service import retrieve_chunks


def format_chunks(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No relevant notes found."
    lines = []
    for chunk in chunks:
        label = chunk.source or "unknown"
        lines.append(f"[{label} | score={chunk.score:.2f}] {chunk.text}")
    return "\n\n".join(lines)


async def search_notes(question: str, *, top_k: int = 5) -> tuple[str, list[RetrievedChunk]]:
    """Tool: только поиск по заметкам (без chat)."""
    chunks = await retrieve_chunks(question, top_k=top_k)
    return format_chunks(chunks), chunks