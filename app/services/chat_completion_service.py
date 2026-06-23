import json
import logging
from bson import ObjectId

from app.llm.factory import get_llm_provider
from app.llm.params import resolve_llm_params, CompletionContext
from app.models.auth_model import MavanUser
from app.models.data_models import ChatCompletionRequest
from app.services.persist_chat import persist_chat, update_chat

logger = logging.getLogger(__name__)


def parse_attachment_ids(header_value: str | None) -> list[str] | None:
    if not header_value or not header_value.strip():
        return None
    return [x.strip() for x in header_value.split(",") if x.strip()]


async def run_persist(
        *,
        current_user: MavanUser,
        chat_request: ChatCompletionRequest,
        attachment_ids: list[str] | None,
        ctx: CompletionContext
) -> tuple[ObjectId, ObjectId]:
    task_type = chat_request.task_type
    if not task_type:
        system = next((m for m in chat_request.messages if m.role == "system"), None)
        task_type = system.name if system else None

    chat_id, message_id, _ = await persist_chat(
        current_user=current_user,
        chat_request=chat_request,
        task_type=task_type,
        model=ctx.model,
        attachment_ids=attachment_ids
    )
    return chat_id, message_id


async def stream_completion_events(
        *,
        chat_request: ChatCompletionRequest,
        chat_id: ObjectId,
        message_id: ObjectId,
        ctx: CompletionContext
):
    """
  Async generator of (event_name, data_str) for SSE.
  Guarantees update_chat in finally on success path.
    """
    provider = get_llm_provider()
    accumulated = ""

    yield "start", json.dumps({"chat_id": str(chat_id), "message_id": str(message_id)})

    try:
        async for delta in provider.stream(
                messages=chat_request.messages,
                **ctx.as_llm_kwargs()
        ):
            accumulated += delta
            yield "partial", accumulated.strip()  # как в f852fd8; можно сменить на delta-only

        yield "completed", accumulated.strip()
        await update_chat(message_id=message_id, content_assistant=accumulated.strip())

    except Exception as e:
        logger.exception("LLM stream failed")
        yield "error", json.dumps({"error": str(e)})
        # опционально: await mark_message_failed(message_id)
        raise
