from app.rag.pipeline import default_pipeline
from app.rag.types import RagAnswer


async def answer_question(question: str, *, provider: str | None = None, top_k: int = 5) -> RagAnswer:
    return await default_pipeline.answer(question, provider_name=provider, top_k=top_k)
