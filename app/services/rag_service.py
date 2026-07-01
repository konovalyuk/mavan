from app.rag.pipeline import get_default_pipeline
from app.rag.types import RagAnswer, RetrievedChunk


async def retrieve_chunks(question: str, *, top_k: int = 5) -> list[RetrievedChunk]:
    return await get_default_pipeline().retrieve_chunks(question, top_k=top_k)


async def answer_question(question: str, *, provider: str | None = None, top_k: int = 5) -> RagAnswer:
    return await get_default_pipeline().answer(question, provider_name=provider, top_k=top_k)
