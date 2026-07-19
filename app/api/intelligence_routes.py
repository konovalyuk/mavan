from fastapi import APIRouter, Body, Depends

from app.domain.schemas import DecisionRequest
from app.models.auth_model import MavanUser
from app.models.chat_model import MavanChatCompletionRequest
from app.services.auth_service import get_user_from_token
from app.services import decision_service
from app.services.conversation_service import handle_turn

router = APIRouter()


@router.post("/api/v1/chat/completions")
async def chat_completions_endpoint(
        chat_request: MavanChatCompletionRequest = Body(...),
        current_user: MavanUser = Depends(get_user_from_token),
):
    return await handle_turn(chat_request=chat_request, current_user=current_user)


@router.post("/api/v1/decisions/recommend")
async def recommend(
        body: DecisionRequest = Body(...),
        current_user: MavanUser = Depends(get_user_from_token),
):
    return await decision_service.recommend(body)
