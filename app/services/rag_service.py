import json
from typing import AsyncIterator
from app.rag.pipeline import get_default_pipeline
from app.rag.types import RagAnswer, RetrievedChunk


async def retrieve_chunks(question: str, *, top_k: int = 5) -> list[RetrievedChunk]:
    return await get_default_pipeline().retrieve_chunks(question, top_k=top_k)


async def answer_question(question: str, *, provider: str | None = None, top_k: int = 5) -> RagAnswer:
    return await get_default_pipeline().answer(question, provider_name=provider, top_k=top_k)


def _serialize_sources(chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "text": c.text,
            "score": c.score,
            "source": c.source,
            "chunk_id": c.chunk_id,
        }
        for c in chunks
    ]


async def stream_rag_events(question: str, *, provider: str | None = None, top_k: int = 5) -> AsyncIterator[tuple[str, str]]:
    accumulated = ""
    sources: list[RetrievedChunk] = []

    yield "start", json.dumps({"question": question})

    try:
        async for delta, chunks in get_default_pipeline().answer_stream(question, provider_name=provider, top_k=top_k):
            sources = chunks
            accumulated += delta
            yield "partial", accumulated.strip()

        yield "completed", json.dumps({
            "answer": accumulated.strip(),
            "sources": _serialize_sources(sources),
        })
    except Exception as e:
        yield "error", json.dumps({"error": str(e)})
        raise
