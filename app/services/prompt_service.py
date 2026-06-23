import logging
from datetime import datetime, timezone
from typing import List

from fastapi import HTTPException
from pymongo import DESCENDING, ReturnDocument
from pymongo.errors import PyMongoError

from app.database import get_db
from app.models.prompt_model import PromptPreview, PromptCreate, PromptResponse
from app.services.util_service import parse_object_id

logger = logging.getLogger(__name__)


async def find_all_system_prompts(except_labels: list[str]) -> List[dict]:
    try:
        prompt_filter = {}
        if except_labels is not None:
            prompt_filter["label"] = {"$nin": except_labels}

        cursor = get_db().prompts.find(prompt_filter).sort("updated_at", DESCENDING)
        prompts = await cursor.to_list(length=None)
        return [
            PromptPreview(**prompt).model_dump(by_alias=False, mode="json")
            for prompt in prompts
        ]
    except Exception as e:
        logger.error("Failed to fetch prompts: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load prompts")


async def create_system_prompt(system_prompt: PromptCreate, username: str) -> PromptResponse:
    try:
        system_prompt_doc = {
            **system_prompt.model_dump(),
            "created_at": datetime.now(timezone.utc),
            "created_by": username
        }
        result = await get_db().prompts.insert_one(system_prompt_doc)
        logger.info("Created new prompt: %s by user=%s", str(result.inserted_id), username)
        return PromptResponse(id=str(result.inserted_id))
    except PyMongoError as e:
        logger.exception("MongoDB error while creating prompt: %s", e)
        raise HTTPException(status_code=500, detail="Database error")
    except Exception as e:
        logger.exception("Unexpected error while creating prompt: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


async def update_system_prompt(prompt_id: str, system_prompt: PromptCreate, username: str) -> PromptResponse:
    try:
        object_id = parse_object_id(prompt_id)
        prompt_filter = {"_id": object_id}
        update_fields = {
            "label": system_prompt.label,
            "icon": system_prompt.icon,
            "content": system_prompt.content,
            "updated_at": datetime.now(timezone.utc),
            "updated_by": username,
        }

        updated = await get_db().prompts.find_one_and_update(
            prompt_filter,
            {"$set": update_fields},
            return_document=ReturnDocument.AFTER,
        )
        if not updated:
            raise HTTPException(status_code=404, detail="Prompt not found or not owned by user")
        return PromptResponse(id=str(updated["_id"]))
    except Exception as e:
        logger.exception("Failed to update prompt %s: %s", prompt_id, e)
        raise


async def delete_system_prompt(prompt_id: str, username: str) -> PromptResponse:
    try:
        object_id = parse_object_id(prompt_id)
        prompt_filter = {"_id": object_id}
        result = await get_db().prompts.delete_one(prompt_filter)
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Prompt not found or not owned by user")
        return PromptResponse(id=prompt_id)
    except Exception as e:
        logger.exception("Failed to delete prompt %s: %s", prompt_id, e)
        raise
