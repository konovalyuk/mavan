import json
import logging
from typing import AsyncIterator
from bson import ObjectId
from config import rag_settings

from sse_starlette.sse import EventSourceResponse, ServerSentEvent
from app.llm.capabilities import Capability, get_capability
from app.llm.chat.chat_providers import prepare_chat_request
from app.llm.chat.schemas import ChatCompletionRequest
from app.models.auth_model import MavanUser
from app.models.data_models import MavanChatCompletionRequest
from app.services.persist_chat import persist_chat, update_chat
from app.rag.pipeline import get_default_pipeline
from app.rag.context_budget import prepare_context_chunks
from app.rag.prompts import build_rag_messages

logger = logging.getLogger(__name__)


def parse_attachment_ids(header_value: str | None) -> list[str] | None:
    if not header_value or not header_value.strip():
        return None
    return [x.strip() for x in header_value.split(",") if x.strip()]


def _resolve_task_type(chat_request: MavanChatCompletionRequest) -> str | None:
    if chat_request.task_type:
        return chat_request.task_type
    system = next((m for m in chat_request.messages if m.role == "system"), None)
    return system.name if system else None


async def _maybe_enrich_with_rag(chat_request: MavanChatCompletionRequest) -> ChatCompletionRequest:
    if not chat_request.use_rag:
        return prepare_chat_request(chat_request, provider=chat_request.provider)

    user_msg = next((m for m in chat_request.messages if m.role == "user" and m.content), None)
    if not user_msg:
        return prepare_chat_request(chat_request, provider=chat_request.provider)

    chunks = await get_default_pipeline().retrieve_chunks(user_msg.content, top_k=chat_request.rag_top_k)
    chunks = prepare_context_chunks(chunks, min_score=rag_settings.MIN_SCORE, max_chars=rag_settings.CONTEXT_MAX_CHARS,
                                    min_relative=rag_settings.MIN_RELATIVE_SCORE)
    rag_messages = build_rag_messages(user_msg.content, chunks)

    return prepare_chat_request(
        ChatCompletionRequest(
            messages=rag_messages,
            model=chat_request.model,
            max_tokens=chat_request.max_tokens,
            temperature=chat_request.temperature,
            stream=chat_request.stream,
        ),
        provider=chat_request.provider,
    )


async def chat_completions(
        *,
        current_user: MavanUser,
        chat_request: MavanChatCompletionRequest,
        attachment_ids: list[str] | None = None,
):
    """Единая точка: persist + stream или complete."""
    provider_name = chat_request.provider
    request = await _maybe_enrich_with_rag(chat_request)

    chat_id, message_id, attachment_ids = await persist_chat(
        current_user=current_user,
        chat_request=chat_request,
        task_type=_resolve_task_type(chat_request),
        model=request.model,
        attachment_ids=attachment_ids
    )

    if chat_request.stream:
        async def event_generator():
            async for event, data in stream_completion_events(
                    request=request,
                    chat_id=chat_id,
                    message_id=message_id,
                    provider_name=provider_name,
            ):
                yield ServerSentEvent(event=event, data=data)

        return EventSourceResponse(event_generator())

    provider = get_capability(Capability.CHAT)(provider_name)
    response = await provider.complete(request)
    await update_chat(message_id=message_id, content_assistant=response.text)

    return {
        "chat_id": str(chat_id),
        "message_id": str(message_id),
        "model": response.model,
        "content": response.text,
    }


async def stream_completion_events(
        *,
        request: ChatCompletionRequest,
        chat_id: ObjectId,
        message_id: ObjectId,
        provider_name: str | None = None
) -> AsyncIterator[tuple[str, str]]:
    provider = get_capability(Capability.CHAT)(provider_name)

    accumulated = ""
    yield "start", json.dumps({"chat_id": str(chat_id), "message_id": str(message_id)})

    try:
        async for delta in provider.stream(request):
            accumulated += delta
            yield "partial", accumulated.strip()

        yield "completed", accumulated.strip()
        await update_chat(message_id=message_id, content_assistant=accumulated.strip())

    except Exception as e:
        logger.exception("LLM stream failed")
        yield "error", json.dumps({"error": str(e)})
        raise
