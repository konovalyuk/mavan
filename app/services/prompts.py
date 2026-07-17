from typing import Literal
from app.rag.types import RetrievedChunk

TASK_INSTRUCTIONS: dict[str, str] = {
    "translate": (
        "You are a professional translator. "
        "Translate the user's text accurately. Preserve meaning, tone, and formatting."
    ),
    "summarize": (
        "You are a concise summarizer. "
        "Produce a clear summary of the user's text."
    ),
    "transcribe": (
        "You are a transcription assistant. "
        "Transcribe or format the provided content accurately."
    ),
    "qa": "Answer the user's question clearly and accurately.",
}

DEFAULT_INSTRUCTION = "You are a helpful assistant."

RAG_GROUNDING = (
    "Answer using the provided context when it is relevant.\n"
    "If the answer is not in the context, respond with: I don't know.\n"
    "Do not use outside knowledge.\n"
    "Each context block has a [source: ...] label. Prefer the highest-scored source."
)

AGENT_ROLE = "You are a helpful assistant with access to tools."

AGENT_TOOL_POLICY = (
    "Use tools when they help answer the user's question.\n"
    "- search_notes: indexed documents / knowledge in the current scope\n"
    "- web_search: current events or facts not in notes\n"
    "- run_python: calculations\n"
    "You may call tools multiple times with different queries.\n"
    "Base answers on tool results when you used tools.\n"
    "If tools do not provide enough information, say you do not know.\n"
    "Do not invent tool results."
)


def format_chunk(chunk: RetrievedChunk) -> str:
    label = chunk.source or "unknown"
    return f"[source: {label} | score: {chunk.score:.2f}]\n{chunk.text}"


def format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(no context provided)"
    return "\n\n---\n\n".join(format_chunk(c) for c in chunks)


def build_system_content(mode: Literal["ask", "agent"] = "ask", task_type: str | None = None, custom_system: str | None = None, chunks: list[RetrievedChunk] | None = None) -> str:
    """Single system message: base (custom|task|default) + agent policy + optional RAG."""
    if custom_system and custom_system.strip():
        parts = [custom_system.strip()]
    elif task_type and task_type in TASK_INSTRUCTIONS:
        parts = [TASK_INSTRUCTIONS[task_type]]
    else:
        parts = [AGENT_ROLE if mode == "agent" else DEFAULT_INSTRUCTION]

    if mode == "agent":
        parts.append(AGENT_TOOL_POLICY)

    if chunks is not None:
        parts.append(RAG_GROUNDING)
        parts.append(f"Context:\n{format_context(chunks)}")

    return "\n\n".join(parts)