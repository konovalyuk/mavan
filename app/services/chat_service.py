from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import DESCENDING, ReturnDocument
from typing import Optional, List, Dict, Any

from fastapi import HTTPException

from app.database import get_db, get_client
from app.models.chat_model import ChatUpdate, ChatBaseResponse, ChatResponse
from app.services.file_service import soft_delete_attachments
from app.services.search_service import search_messages_by_text, search_chats_by_title, build_space_insensitive_regex
from app.services.util_service import parse_object_id
import logging
import re

logger = logging.getLogger(__name__)


async def find_paginated_chats_for_user(username: str, limit: int, before: str | None, project_id: str | None = None) -> \
List[ChatBaseResponse]:
    """
    Returns a paginated list of the user's chats, sorted by updated_at DESC.
    - If `project_id` is not specified: only return chats that have an empty project_id.
    - If `project_id` is specified: only return chats in this project.
    """
    try:
        query: dict = {"created_by": username}

        if before:
            try:
                before_dt = datetime.fromisoformat(before.replace("Z", "+00:00"))
                query["updated_at"] = {"$lt": before_dt}
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid 'before' datetime format")

        if project_id:
            object_id = parse_object_id(project_id)
            project_filter = {"_id": object_id, "created_by": username}
            project = await get_db().projects.find_one(project_filter)
            if not project:
                raise HTTPException(status_code=404,
                                    detail=f"Project with id {project_id} not found or not owned by the user")

            query["project_id"] = project_id
        else:
            query["project_id"] = {"$exists": False}

        cursor = get_db().chats.find(
            query,
            projection={"_id": 1, "project_id": 1, "title": 1, "updated_at": 1}
        ).sort("updated_at", DESCENDING).limit(limit)

        chat_docs = await cursor.to_list(length=limit)

        return [
            ChatBaseResponse.from_mongo(chat_doc)
            for chat_doc in chat_docs
        ]

    except Exception as e:
        logger.error("Failed to paginate chats for user %s: %s", username, str(e), exc_info=True)
        raise


async def find_chat_by_id(username: str, chat_id: str) -> ChatResponse:
    """
    Find a single chat by its ID'.
    Raises:
        HTTPException 400: Invalid chat ID format.
        HTTPException 404: Chat not found.
    """
    try:
        object_id = parse_object_id(chat_id)
        chat_filter = {"_id": object_id, "created_by": username}

        chat_dict = await get_db().chats.find_one(chat_filter)
        if not chat_dict:
            raise HTTPException(status_code=404, detail=f"Chat with id {chat_id} not found")

        return ChatResponse.from_mongo(chat_dict)
    except Exception as e:
        logger.error(
            "Failed to find chat (chat_id=%s) for user %s: %s",
            chat_id, username, str(e), exc_info=True
        )
        raise


async def rename_chat(chat_id: str, update_data: ChatUpdate, username: str) -> ChatResponse:
    try:
        object_id = parse_object_id(chat_id)
        chat_filter = {"_id": object_id, "created_by": username}
        update_fields = {
            "title": update_data.title,
            "updated_at": datetime.now(timezone.utc)
        }
        updated_doc = await get_db().chats.find_one_and_update(
            chat_filter,
            {"$set": update_fields},
            return_document=ReturnDocument.AFTER
        )

        if not updated_doc:
            raise HTTPException(status_code=404, detail="Chat not found or not owned by user")
        return ChatResponse.from_mongo(updated_doc)
    except Exception as e:
        logger.error("Failed to rename chat %s: %s", chat_id, str(e), exc_info=True)
        raise


async def delete_chat_and_messages(chat_id: str, username: str) -> None:
    """
    Soft delete a chat, messages, and attachments using MongoDB transactions.
    Moves records to *_deleted collections instead of hard deleting.
    """
    try:
        client = get_client()
        db = get_db()

        async with await client.start_session() as session:
            async with session.start_transaction():
                await soft_delete_chat(db, session, chat_id, username)
                await soft_delete_messages(db, session, chat_id, username)
                await soft_delete_attachments(db, session, chat_id, username)
    except Exception as e:
        logger.error("Failed to delete chat %s: %s", chat_id, str(e), exc_info=True)
        raise


async def soft_delete_chat(db, session, chat_id: str, username: str):
    """Soft delete chat by moving to chats_deleted collection."""
    try:
        object_id = parse_object_id(chat_id)
        chat_filter = {"_id": object_id, "created_by": username}
        chat_dict = await db.chats.find_one(chat_filter, session=session)
        if not chat_dict:
            raise HTTPException(status_code=404, detail="Chat not found or not owned by user")
        await db.chats_deleted.insert_one(chat_dict, session=session)
        await db.chats.delete_one(chat_filter, session=session)
        logger.info("Soft deleted chat %s from collection chats", chat_id)
    except Exception as e:
        logger.error("Failed to soft delete chat %s: %s", chat_id, str(e), exc_info=True)
        raise


async def soft_delete_messages(db, session, chat_id: str, username: str):
    """Soft delete messages by moving to messages_deleted collection."""
    try:
        message_filter = {"chat_id": chat_id, "created_by": username}
        cursor = db.messages.find(message_filter, session=session)
        message_dict = await cursor.to_list(length=None)
        if not message_dict:
            raise HTTPException(status_code=404, detail="Messages are not found or not owned by user")
        await db.messages_deleted.insert_many(message_dict, session=session)
        await db.messages.delete_many(message_filter, session=session)
        logger.info("Delete messages for chat %s from collection messages", chat_id)
    except Exception as e:
        logger.error("Failed to soft delete messages for chat %s: %s", chat_id, str(e), exc_info=True)
        raise


async def search_chat_message(text: str, search_in: List[str], skip: int, limit: int, username: str) -> List[
    Dict[str, Any]]:
    if not text.strip():
        logger.warning("text is empty, nothing to search")
        raise HTTPException(status_code=400, detail="Search text must not be empty")

    if not search_in:
        logger.warning("search_in is empty, nothing to search")
        raise HTTPException(status_code=400, detail="Parameter 'search_in' cannot be empty")

    try:
        db: AsyncIOMotorDatabase = get_db()
        normalized_regex = build_space_insensitive_regex(text.strip())
        pattern = re.compile(normalized_regex, re.IGNORECASE)

        message_results: List[Dict[str, Any]] = []
        title_results: List[Dict[str, Any]] = []

        chat_ids_found = set()

        if "content_user" in search_in or "content_assistant" in search_in:
            message_results = await search_messages_by_text(db, normalized_regex, pattern, search_in, username)
            chat_ids_found.update([res["chat_id"] for res in message_results])

        if "title" in search_in:
            title_results = await search_chats_by_title(db, normalized_regex, pattern, username,
                                                        exclude_ids=chat_ids_found)

        combined = sorted(message_results + title_results, key=lambda x: x["updated_at"], reverse=True)
        return combined[skip:skip + limit]
    except Exception as e:
        logger.exception("Unexpected error in search_text: %s", str(e))
        raise


async def verify_chat_exists(chat_id: str, username: str):
    """Raise 404 if chat not found or does not belong to username."""
    object_id = parse_object_id(chat_id)
    chat_filter = {"_id": object_id, "created_by": username}
    chat_dict = await get_db().chats.find_one(chat_filter, projection={"_id": 1})
    if not chat_dict:
        raise HTTPException(status_code=404, detail=f"Chat with id {chat_id} not found")
