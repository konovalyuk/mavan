from fastapi import APIRouter, Body, Depends, Header
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from app.models.auth_model import MavanUser
from app.models.data_models import ChatCompletionRequest
from app.services.auth_service import get_user_from_token
from app.services.chat_completion_service import (
    parse_attachment_ids,
    run_persist,
    stream_completion_events,
)

# + non-streaming helper если stream=false

router = APIRouter()


@router.post("/api/v1/chat/completions")
async def chat_completions(
        chat_request: ChatCompletionRequest = Body(...),
        attachment_ids_header: str | None = Header(None, alias="attachment-ids"),
        current_user: MavanUser = Depends(get_user_from_token),
):
    attachment_ids = parse_attachment_ids(attachment_ids_header)
    chat_id, message_id = await run_persist(
        current_user=current_user,
        chat_request=chat_request,
        attachment_ids=attachment_ids,
    )

    if chat_request.stream:
        async def event_generator():
            async for event, data in stream_completion_events(
                    chat_request=chat_request,
                    chat_id=chat_id,
                    message_id=message_id,
            ):
                yield ServerSentEvent(event=event, data=data)

        return EventSourceResponse(event_generator())

    # non-streaming — отдельная ветка: собрать full text, update_chat, вернуть ChatCompletionResponse
