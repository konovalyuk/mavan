from app.services.rag_service import answer_question


async def search_notes(question: str, *, provider: str | None = None) -> str:
    """Tool: поиск по заметкам через RAG."""
    result = await answer_question(question, provider=provider, top_k=5)
    if not result.sources:
        return "No relevant notes found."

    lines = []
    for chunk in result.sources:
        label = chunk.source or "unknown"
        lines.append(f"[{label} | score={chunk.score:.2f}] {chunk.text}")
    return "\n\n".join(lines)
