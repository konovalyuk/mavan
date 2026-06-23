# stdlib
from typing import List

# third-party
from fastapi import APIRouter, Depends, Path, Body
from fastapi.params import Query

# internal/local
from app.services.auth_service import get_user_from_token
from app.models.auth_model import MavanUser
from app.models.prompt_model import PromptCreate, PromptResponse
from app.services.prompt_service import (
    find_all_system_prompts,
    create_system_prompt,
    update_system_prompt,
    delete_system_prompt,
)

router = APIRouter()


@router.get("/api/v1/prompts", summary="Get all prompts", response_model=List[dict],
            responses={400: {"description": "Invalid data"}, 500: {"description": "Internal Server Error"}})
async def api_find_all_system_prompts(
        except_labels: list[str] | None = Query(None, alias="except_label",
                                                description="Comma-separated labels to exclude from results"),
        current_user: MavanUser = Depends(get_user_from_token)
) -> List[dict]:
    return await find_all_system_prompts(except_labels)


@router.post("/api/v1/prompts", summary="Create system prompt", response_model=PromptResponse,
             responses={400: {"description": "Invalid data"}, 500: {"description": "Internal Server Error"}},
             dependencies=[Depends(get_user_from_token)])
async def api_create_system_prompt(
        system_prompt: PromptCreate = Body(..., description="New system prompt"),
        current_user: MavanUser = Depends(get_user_from_token)
):
    return await create_system_prompt(system_prompt, current_user.username)


@router.put("/api/v1/prompts/{prompt_id}", summary="Update system prompt", response_model=PromptResponse,
            responses={400: {"description": "Invalid data"}, 500: {"description": "Internal Server Error"}},
            dependencies=[Depends(get_user_from_token)])
async def api_update_system_prompt(
        prompt_id: str = Path(..., description="ID of the prompt"),
        system_prompt: PromptCreate = Body(..., description="New system prompt"),
        current_user: MavanUser = Depends(get_user_from_token)
):
    return await update_system_prompt(prompt_id, system_prompt, current_user.username)


@router.delete("/api/v1/prompts/{prompt_id}", summary="Delete system prompt", response_model=PromptResponse,
               responses={400: {"description": "Invalid data"}, 500: {"description": "Internal Server Error"}},
               dependencies=[Depends(get_user_from_token)])
async def api_delete_system_prompt(
        prompt_id: str = Path(..., description="ID of the prompt"),
        current_user: MavanUser = Depends(get_user_from_token)
):
    return await delete_system_prompt(prompt_id, current_user.username)
