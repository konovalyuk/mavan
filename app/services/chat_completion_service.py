import json
import logging
from bson import ObjectId

from app.llm.capabilities import Capability, get_capability
from app.llm.chat.chat_providers import build_chat_request
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
        provider_name: str | None = None
) -> tuple[ObjectId, ObjectId]:
    task_type = chat_request.task_type
    if not task_type:
        system = next((m for m in chat_request.messages if m.role == "system"), None)
        task_type = system.name if system else None

    request = build_chat_request(
        messages=chat_request.messages,
        request_model=chat_request.model,
        max_tokens=chat_request.max_tokens,
        temperature=chat_request.temperature,
        provider=provider_name,          # один provider_name
    )

    chat_id, message_id, _ = await persist_chat(
        current_user=current_user,
        chat_request=chat_request,
        task_type=task_type,
        model=request.model,
        attachment_ids=attachment_ids
    )
    return chat_id, message_id


async def stream_completion_events(
        *,
        chat_request: ChatCompletionRequest,
        chat_id: ObjectId,
        message_id: ObjectId,
        provider_name: str | None = None
):
    provider = get_capability(Capability.CHAT)(provider_name)

    request = build_chat_request(
        messages=chat_request.messages,
        request_model=chat_request.model,
        max_tokens=chat_request.max_tokens,
        temperature=chat_request.temperature,
        provider=provider_name,          # один provider_name
    )

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