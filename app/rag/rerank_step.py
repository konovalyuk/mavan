from config import rag_settings
from app.llm.capabilities import Capability, get_capability
from app.llm.rerank.rerank_providers import resolve_rerank_config
from app.rag.types import RetrievedChunk


async def maybe_rerank(
        question: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int,
) -> list[RetrievedChunk]:
    if not rag_settings.RERANK_ENABLED or not chunks:
        return chunks[:top_k]

    cfg = resolve_rerank_config()
    reranker = get_capability(Capability.RERANK)(cfg.provider)
    return await reranker.rerank(question, chunks, model=cfg.model, top_k=top_k)
