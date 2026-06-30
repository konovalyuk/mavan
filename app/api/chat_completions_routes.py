from fastapi import APIRouter, Body, Depends, Header

from app.models.auth_model import MavanUser
from app.models.data_models import MavanChatCompletionRequest
from app.services.auth_service import get_user_from_token
from app.services.chat_completion_service import chat_completions, parse_attachment_ids

router = APIRouter()


@router.post("/api/v1/chat/completions")
async def chat_completions_endpoint(
        chat_request: MavanChatCompletionRequest = Body(...),
        attachment_ids_header: str | None = Header(None, alias="attachment-ids"),
        current_user: MavanUser = Depends(get_user_from_token),
):
    return await chat_completions(
        current_user=current_user,
        chat_request=chat_request,
        attachment_ids=parse_attachment_ids(attachment_ids_header),
    )
