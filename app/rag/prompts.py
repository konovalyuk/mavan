from app.llm.chat.schemas import ChatMessage
from app.rag.types import RetrievedChunk

SYSTEM_INSTRUCTION = (
    "Answer only using the provided context.\n"
    "If the answer is not in the context, respond with: I don't know.\n"
    "Do not use outside knowledge.\n"
    "Each context block has a [source: ...] label. Prefer the highest-scored source."
)


def format_chunk(chunk: RetrievedChunk) -> str:
    label = chunk.source or "unknown"
    return f"[source: {label} | score: {chunk.score:.2f}]\n{chunk.text}"


def format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(no context provided)"
    return "\n\n---\n\n".join(format_chunk(c) for c in chunks)


def build_rag_messages(question: str, chunks: list[RetrievedChunk]) -> list[ChatMessage]:
    context = format_context(chunks)
    return [
        ChatMessage(
            role="system",
            content=f"{SYSTEM_INSTRUCTION}\n\nContext:\n{context}",
        ),
        ChatMessage(role="user", content=question),
    ]