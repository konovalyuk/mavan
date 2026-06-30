from dataclasses import dataclass
from pathlib import Path

from config import rag_settings
from app.llm.capabilities import Capability, get_capability
from app.llm.chat.chat_providers import prepare_chat_request
from app.llm.chat.schemas import ChatCompletionRequest
from app.rag.context_budget import prepare_context_chunks
from app.rag.prompts import build_rag_messages
from app.rag.retriever import Retriever
from app.rag.retrievers.file import FileRetriever
from app.rag.retrievers.vector import VectorRetriever
from app.rag.types import RagAnswer
from app.rag.rerank_step import maybe_rerank

RAG_TEMPERATURE = 0.2
DEFAULT_TOP_K = 5


def build_retriever() -> Retriever:
    kind = rag_settings.RETRIEVER.lower()
    if kind == "file":
        return FileRetriever(Path(rag_settings.DATA_DIR), index_path=Path(rag_settings.INDEX_PATH))
    if kind == "vector":
        return VectorRetriever(Path(rag_settings.VECTOR_INDEX_PATH))
    raise ValueError(f"Unknown RAG_RETRIEVER: {kind!r}. Use 'vector' or 'file'.")


@dataclass
class RagPipeline:
    retriever: Retriever

    async def answer(self, question: str, *, provider_name: str | None = None, top_k: int = DEFAULT_TOP_K,
                     stream: bool = False) -> RagAnswer:
        fetch_k = rag_settings.FETCH_K if rag_settings.RERANK_ENABLED else top_k
        raw = await self.retriever.retrieve(question, top_k=fetch_k)
        raw = await maybe_rerank(question, raw, top_k=top_k)
        chunks = prepare_context_chunks(raw, min_score=rag_settings.MIN_SCORE, max_chars=rag_settings.CONTEXT_MAX_CHARS, min_relative=rag_settings.MIN_RELATIVE_SCORE)
        messages = build_rag_messages(question, chunks)
        request = prepare_chat_request(ChatCompletionRequest(messages=messages, temperature=RAG_TEMPERATURE), provider=provider_name)
        provider = get_capability(Capability.CHAT)(provider_name)
        if stream:
            raise NotImplementedError("use answer_stream()")
        response = await provider.complete(request)
        return RagAnswer(text=response.text, sources=chunks)

    async def answer_stream(self, question: str, *, provider_name: str | None = None, top_k: int = DEFAULT_TOP_K):
        fetch_k = rag_settings.FETCH_K if rag_settings.RERANK_ENABLED else top_k
        raw = await self.retriever.retrieve(question, top_k=fetch_k)
        raw = await maybe_rerank(question, raw, top_k=top_k)
        chunks = prepare_context_chunks(raw, min_score=rag_settings.MIN_SCORE, max_chars=rag_settings.CONTEXT_MAX_CHARS)
        messages = build_rag_messages(question, chunks)
        request = prepare_chat_request(ChatCompletionRequest(messages=messages, temperature=RAG_TEMPERATURE), provider=provider_name)
        provider = get_capability(Capability.CHAT)(provider_name)
        async for delta in provider.stream(request):
            yield delta, chunks


default_pipeline = RagPipeline(retriever=build_retriever())


async def answer(question: str, **kwargs) -> RagAnswer:
    return await default_pipeline.answer(question, **kwargs)
