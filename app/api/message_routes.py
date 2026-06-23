# stdlib
from typing import Literal, Optional

# third-party
from fastapi import APIRouter, Depends, Path, Body, Query, status
from pydantic import constr

from app.models.message_model import MessageReactionResponse, MessageResponse, ActivateChildRequest, \
    ActivateChildResponse
# internal/local
from app.services.auth_service import get_user_from_token
from app.models.auth_model import MavanUser
from app.services.message_service import load_chat_messages, add_reaction_to_message, activate_message_branch

router = APIRouter()


@router.get("/api/v1/chats/{chat_id}/messages", summary="Get messages from chat with token-based pagination",
            status_code=status.HTTP_200_OK,
            response_model=list[MessageResponse],
            response_model_exclude_none=True,
            responses={400: {"description": "Bad request"},
                       403: {"description": "Access denied"},
                       404: {"description": "Not found"},
                       500: {"description": "Internal Server Error"}})
async def api_load_chat_messages(
        chat_id: str,
        limit: int = Query(10, ge=1, le=100),
        parent_message_id: str | None = Query(None, description="ID of the parent message to get replies for"),
        current_user: MavanUser = Depends(get_user_from_token)
) -> list[MessageResponse]:
    return await load_chat_messages(chat_id, current_user.username, limit, parent_message_id)


@router.patch("/api/v1/chats/{chat_id}/messages/{message_id}/reaction",
              summary="Add or update reaction to the message in the chat",
              status_code=status.HTTP_200_OK,
              responses={400: {"description": "Bad request"},
                         403: {"description": "Access denied"},
                         404: {"description": "Chat or message not found"},
                         500: {"description": "Internal Server Error"}})
async def api_add_reaction_to_message(
        chat_id: str = Path(..., description="ID of the chat"),
        message_id: str = Path(..., description="ID of the message to react to"),
        reaction: Optional[Literal["like", "dislike", "neutral"]] = Body(
            None, embed=True, description="Reaction type (optional if you only add comment to an existing like)"),
        reaction_comment: Optional[constr(strip_whitespace=True, min_length=1, max_length=500)] = Body(
            None, embed=True, description="Optional reaction comment (required when reaction==dislike)"),
        current_user: MavanUser = Depends(get_user_from_token)
) -> MessageReactionResponse:
    return await add_reaction_to_message(chat_id, message_id, reaction, reaction_comment, current_user.username)


@router.patch("/api/v1/chats/{chat_id}/messages/active-child",
              summary="Switch active message using sibling position",
              status_code=status.HTTP_200_OK,
              response_model=ActivateChildResponse,
              response_model_exclude_none=True,
              responses={400: {"description": "Bad request"},
                         403: {"description": "Access denied"},
                         404: {"description": "Not found"}})
async def api_activate_message_branch(
        chat_id: str = Path(..., description="ID of the chat"),
        parent_message_id: str | None = Query(None, description="ID of the parent message to get replies for"),
        sibling: ActivateChildRequest = Body(..., description="Child message ID to switch to"),
        current_user: MavanUser = Depends(get_user_from_token)
) -> ActivateChildResponse:
    return await activate_message_branch(chat_id, sibling.sibling_position, current_user.username, parent_message_id)
