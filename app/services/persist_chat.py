import logging
from datetime import datetime, timezone
from pymongo import ReturnDocument
from typing import Optional, List, Dict, Any

from fastapi import Request, HTTPException
from bson import ObjectId
from pymongo.errors import PyMongoError

from app.models.auth_model import MavanUser
from app.database import get_db, get_client
from app.models.chat_model import ChatModel
from app.models.message_model import MessageCreate
from app.models.data_models import MavanChatCompletionRequest, ChatMessage
from app.services.chat_service import parse_object_id

logger = logging.getLogger(__name__)


async def update_chat(
        message_id: ObjectId,
        content_assistant: str,
) -> None:
    try:
        db = get_db()
        update_data = {
            "content_assistant": content_assistant,
            "updated_at": datetime.now(timezone.utc)
        }
        result = await db.messages.update_one(
            {"_id": message_id},
            {"$set": update_data}
        )

        if result.matched_count == 0:
            logger.warning("No message found with id: %s", message_id)
        else:
            logger.info("Updated content_assistant for message id: %s", message_id)

    except PyMongoError as e:
        logger.exception("Error updating content_assistant for message %s: %s", message_id, e)
        raise


async def check_and_update_attachments(db, session, attachment_ids: list[str], username: str,
                                       chat_id: ObjectId, message_id: ObjectId) -> None:
    for attachment_id in attachment_ids:
        object_id = parse_object_id(attachment_id)
        attachment_filter = {"_id": object_id, "created_by": username}
        update_query = {
            "$set": {
                "chat_id": str(chat_id),
                "updated_at": datetime.now(timezone.utc),
                "updated_by": username
            },
            "$addToSet": {
                "message_ids": str(message_id)
            }
        }
        existing_attachment = await db.attachments.find_one_and_update(
            attachment_filter,
            update_query,
            return_document=ReturnDocument.AFTER,
            session=session
        )
        if not existing_attachment:
            logger.warning("Attachment with id=%s not found", attachment_id)
            raise HTTPException(status_code=404, detail=f"Attachment with id {attachment_id} not found")


async def update_persist_chat(db, attachment_ids: list[str], now: datetime, username: str, chat_id: ObjectId,
                              message_id: ObjectId) -> None:
    update_fields = {
        "active_message_id": str(message_id),
        "updated_at": now,
        "updated_by": username
    }
    update_query = {"$set": update_fields}
    if attachment_ids:
        update_query["$addToSet"] = {"attachment_ids": {"$each": attachment_ids}}
    await db.chats.update_one(
        {"_id": chat_id},
        update_query
    )


async def persist_chat(
        current_user: MavanUser,
        chat_request: MavanChatCompletionRequest,
        task_type: Optional[str],
        model: str,
        attachment_ids: Optional[list[str]] = None
) -> tuple[ObjectId, ObjectId, Optional[list[str]]]:
    user_message = extract_message(chat_request, "user")
    if not user_message or not user_message.content or not user_message.content.strip():
        raise HTTPException(status_code=400, detail="User message content cannot be empty")
    system_message = extract_message(chat_request, "system")

    client = get_client()
    db = get_db()
    now = datetime.now(timezone.utc)

    try:
        async with await client.start_session() as session:
            async with session.start_transaction():
                project_id = None
                if getattr(chat_request, "project_id", None):
                    project_id = await get_and_update_project(db, session, getattr(chat_request, "project_id", None),
                                                              current_user.username, now)
                chat_id = await get_or_create_chat(db, session, getattr(chat_request, "chat_id", None), model,
                                                   project_id, current_user,
                                                   user_message.content, now)
                message_id, attachment_ids = await insert_user_message(db, session, chat_id,
                                                                       getattr(chat_request, "parent_message_id", None),
                                                                       task_type, current_user,
                                                                       system_message.content if system_message else None,
                                                                       user_message.content, now, attachment_ids)
                if attachment_ids:
                    await check_and_update_attachments(db, session, attachment_ids, current_user.username, chat_id,
                                                       message_id)

        await update_persist_chat(db, attachment_ids, now, current_user.username, chat_id, message_id)
    except Exception as e:
        logger.exception("Transaction failed during persist_chat: %s", e)
        raise

    logger.info("Message has been persist: user=%s, chat_id=%s", current_user.username, chat_id)
    return chat_id, message_id, attachment_ids


async def get_and_update_project(db, session, provided_project_id, username, now) -> str:
    try:
        project_object_id = ObjectId(provided_project_id)
    except Exception:
        logger.warning("Invalid project_id format: %s. Creating new project.", provided_project_id)
        raise HTTPException(status_code=400, detail="Invalid project_id format")

    existing_project = await db.projects.find_one({"_id": project_object_id}, session=session)
    if not existing_project:
        logger.warning("Project with id=%s not found", provided_project_id)
        raise HTTPException(status_code=404, detail="Project with provided project_id not found")

    if existing_project["created_by"] != username:
        raise HTTPException(status_code=403, detail="Access to this project is forbidden")

    return provided_project_id


async def get_or_create_chat(db, session, chat_id, model, project_id, current_user, content_user, now) -> ObjectId:
    if chat_id:
        chat_object_id = await validate_existing_chat(db, session, chat_id, project_id, current_user)

        return chat_object_id

    title = generate_chat_title(content_user)
    chat_doc = ChatModel(
        project_id=project_id,
        title=title,
        model=model,
        root_count=0,
        created_at=now,
        created_by=current_user.username,
        updated_at=now,  # Initialize with creation time
        updated_by=current_user.username  # Initialize with creator
    ).model_dump(exclude_none=True)

    try:
        result = await db.chats.insert_one(chat_doc, session=session)
        logger.info("Created new chat: title=%s, user=%s", title, current_user.username)
        return result.inserted_id
    except PyMongoError as e:
        logger.exception("Error inserting new chat: %s", e)
        raise


async def validate_existing_chat(db, session, provided_chat_id: str, provided_project_id: str,
                                 current_user) -> ObjectId:
    try:
        chat_object_id = ObjectId(provided_chat_id)
    except Exception:
        logger.warning("Invalid chat_id format: %s. Creating new chat.", provided_chat_id)
        raise HTTPException(status_code=400, detail="Invalid chat_id format")

    existing_chat = await db.chats.find_one({"_id": chat_object_id}, session=session)
    if not existing_chat:
        logger.warning("Chat with id=%s not found", provided_chat_id)
        raise HTTPException(status_code=404, detail="Chat with provided chat_id not found")

    if existing_chat.get("project_id") is None and provided_project_id is not None:
        logger.warning("Chat id=%s does not belong to any project, but request was made with project id=%s",
                       provided_chat_id, provided_project_id)
        raise HTTPException(status_code=403, detail="Cannot access a non-project chat when a project_id is provided")

    if existing_chat.get("project_id") is not None and existing_chat.get("project_id") != provided_project_id:
        logger.warning(
            "Chat id=%s belongs to project id=%s, but request was made with project id=%s",
            provided_chat_id, existing_chat.get("project_id"), provided_project_id)
        raise HTTPException(status_code=403, detail="Chat does not belong to the specified project")

    if existing_chat["created_by"] != current_user.username:
        raise HTTPException(status_code=403, detail="Access to this chat is forbidden")

    if existing_chat.get("active_message_id") is None:
        raise HTTPException(status_code=400, detail="Old chats are not supported. Please start a new chat to continue.")

    return chat_object_id


async def insert_user_message(db, session, chat_id, parent_message_id, task_type, current_user, content_system,
                              content_user, now,
                              attachment_ids: Optional[list[str]] = None) -> tuple[ObjectId, Optional[list[str]]]:
    parent_oid = parse_object_id(parent_message_id)
    node_path = None
    if parent_message_id:
        parent_message_doc = await db.messages.find_one_and_update(
            {"_id": parent_oid},
            {"$inc": {"children_count": 1}},
            return_document=ReturnDocument.AFTER,
            session=session
        )
        await validate_existing_message(chat_id, parent_message_doc, current_user)

        parent_node_path = parent_message_doc["node_path"]
        child_number = parent_message_doc["children_count"]
        node_path = f"{parent_node_path}.{child_number}"

        # Inherit attachment_ids from first sibling when regenerating (same question, no new attachments)
        if not attachment_ids and child_number > 1:
            first_sibling = await db.messages.find_one(
                {"parent_message_id": parent_oid, "node_path": f"{parent_node_path}.1"},
                {"attachment_ids": 1, "content_user": 1},
                session=session
            )
            if first_sibling and first_sibling.get("content_user") == content_user and first_sibling.get(
                    "attachment_ids"):
                attachment_ids = first_sibling["attachment_ids"]
                logger.info("Inherited %d attachment_ids from first sibling of parent %s", len(attachment_ids),
                            parent_message_id)
    else:
        chat_doc = await db.chats.find_one_and_update(
            {"_id": chat_id},
            {"$inc": {"root_count": 1}},
            return_document=ReturnDocument.AFTER,
            session=session
        )
        if not chat_doc:
            raise HTTPException(status_code=404, detail="Chat not found")

        root_number = chat_doc["root_count"]
        node_path = str(root_number)

        # Inherit attachment_ids from first root message when regenerating (same question, no new attachments)
        if not attachment_ids and root_number > 1:
            first_root = await db.messages.find_one(
                {"chat_id": str(chat_id), "parent_message_id": None, "node_path": "1"},
                {"attachment_ids": 1, "content_user": 1},
                session=session
            )
            if first_root and first_root.get("content_user") == content_user and first_root.get("attachment_ids"):
                attachment_ids = first_root["attachment_ids"]
                logger.info("Inherited %d attachment_ids from first root message in chat %s", len(attachment_ids),
                            chat_id)

    message_doc = MessageCreate(
        chat_id=str(chat_id),
        attachment_ids=attachment_ids or None,
        task_type=task_type,
        content_system=content_system,
        content_user=content_user,
        content_assistant=None,
        children_count=0,
        node_path=node_path,
        created_at=now,
        created_by=current_user.username
    ).model_dump(exclude_none=True)
    message_doc.update(parent_message_id=parent_oid) if parent_oid else None
    try:
        result = await db.messages.insert_one(message_doc, session=session)
        logger.info("Saved user message in chat %s", chat_id)
        return result.inserted_id, attachment_ids
    except PyMongoError as e:
        logger.exception("Error saving message: %s", e)
        raise


async def validate_existing_message(provided_chat_id: str, parent_message_doc, current_user):
    if not parent_message_doc:
        raise HTTPException(status_code=404, detail="Message with parent_message_id not found")

    if parent_message_doc["created_by"] != current_user.username:
        raise HTTPException(status_code=403, detail="Access to this message is forbidden")

    if parent_message_doc["chat_id"] != str(provided_chat_id):
        raise HTTPException(status_code=403, detail="Access to this chat is forbidden")


def extract_message(chat_request: MavanChatCompletionRequest, role: str) -> Optional[ChatMessage]:
    return next(
        (
            msg for msg in chat_request.messages
            if msg.role == role and isinstance(msg.content, str) and msg.content.strip()
        ),
        None
    )


def generate_chat_title(text: str, max_chars: int = 30) -> str:
    return text[:max_chars] + "..." if len(text) > max_chars else text
