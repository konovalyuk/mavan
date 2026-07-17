import logging
from datetime import datetime, timezone
from pymongo import ReturnDocument
from typing import Optional

from fastapi import HTTPException
from bson import ObjectId
from pymongo.errors import PyMongoError

from app.agents.types import AGENT_TOOL_CHOICE, AGENT_TEMPERATURE
from app.llm.chat.schemas import ChatMessage
from app.models.auth_model import MavanUser
from app.database import get_db, get_client
from app.models.chat_model import ChatModel, MavanChatCompletionRequest
from app.models.message_model import MessageCreate
from app.services.chat_service import parse_object_id
from app.services.message_service import load_message_ancestor_chain

from app.rag.sources import source_prefixes_from_attachment_ids
from app.rag.pipeline import get_default_pipeline
from app.services.prompts import build_system_content
from app.agents.tool_loop.loop import resolve_tools
from app.agents.helpers import chunks_to_sources

logger = logging.getLogger(__name__)


async def save_assistant_reply(message_id: str, content_assistant: str | None = None, mode: str | None = None,
                               sources: list[dict] | None = None, tool_log: list[dict] | None = None, model: str | None = None) -> None:
    try:
        message_object_id = parse_object_id(message_id)
        db = get_db()
        update_data = {
            "content_assistant": content_assistant,
            "updated_at": datetime.now(timezone.utc),
            "mode": mode,
            "sources": sources or [],
            "tool_log": tool_log or [],
            "model": model
        }
        result = await db.messages.update_one(
            {"_id": message_object_id},
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
        if existing_attachment.get("state") != "indexed":
            raise HTTPException(status_code=400, detail=(f"Attachment {attachment_id} is not indexed. Call POST /api/v1/files/index first."))


async def update_chat_meta(db, attachment_ids: list[str] | None, now: datetime, username: str, chat_id: ObjectId,
                           message_id: ObjectId) -> list[str]:
    update_fields = {
        "active_message_id": str(message_id),
        "updated_at": now,
        "updated_by": username
    }
    update_query = {"$set": update_fields}
    if attachment_ids:
        update_query["$addToSet"] = {"attachment_ids": {"$each": attachment_ids}}
    chat_doc = await db.chats.find_one_and_update(
        {"_id": chat_id},
        update_query,
        projection={"attachment_ids": 1},
        return_document=ReturnDocument.AFTER
    )
    return list(chat_doc.get("attachment_ids") or [])


async def save_user_turn(chat_request: MavanChatCompletionRequest, current_user: MavanUser, current_system_message: str | None, current_user_message: str | None) -> tuple[str, str, list[str]]:
    client = get_client()
    db = get_db()
    now = datetime.now(timezone.utc)

    try:
        async with await client.start_session() as session:
            async with session.start_transaction():
                project_id = None
                if chat_request.project_id:
                    project_id = await get_and_update_project(db, session, chat_request.project_id, current_user.username, now)
                chat_id = await get_or_create_chat(db, session, chat_request.chat_id, project_id, current_user, current_user_message, now)
                message_id, attachment_ids = await insert_user_message(db, session, chat_id, chat_request.parent_message_id,
                                                                       chat_request.task_type, current_user, current_system_message,
                                                                       current_user_message, now, chat_request.attachment_ids)
                if attachment_ids:
                    await check_and_update_attachments(db, session, attachment_ids, current_user.username, chat_id, message_id)

        chat_attachment_ids: list[str] = await update_chat_meta(db, attachment_ids, now, current_user.username, chat_id, message_id)
    except Exception as e:
        logger.exception("Transaction failed during prepare_chat_turn: %s", e)
        raise

    logger.info("Message persisted: user=%s, chat_id=%s", current_user.username, chat_id)
    return str(chat_id), str(message_id), chat_attachment_ids


async def prepare_turn(chat_request: MavanChatCompletionRequest, current_user: MavanUser)-> tuple[MavanChatCompletionRequest, str, str, tuple[str, ...] | None]:
    current_system_message: str | None = extract_message(chat_request, "system")
    current_user_message: str | None = extract_message(chat_request, "user")
    if not current_user_message:
        raise HTTPException(400, "User message content cannot be empty")

    history_messages: list[ChatMessage] = []
    if chat_request.parent_message_id:
        if not chat_request.chat_id:
            raise HTTPException(status_code=400, detail="chat_id required when continuing a chat")
        history_messages = await load_chat_history_for_llm(chat_request.chat_id, chat_request.parent_message_id, chat_request.history_limit)

    chat_id, message_id, chat_attachment_ids = await save_user_turn(chat_request, current_user, current_system_message, current_user_message)

    prefixes = source_prefixes_from_attachment_ids(chat_attachment_ids) if chat_attachment_ids else None

    if chat_request.mode == "ask":
        chunks = []
        if chat_attachment_ids:
            chunks = await get_default_pipeline().retrieve_chunks(current_user_message, top_k=chat_request.rag_top_k, source_prefixes=prefixes)
        chat_request.sources = chunks_to_sources(chunks)
        system_content = build_system_content(mode=chat_request.mode, task_type=chat_request.task_type, custom_system=current_system_message, chunks=chunks if chat_attachment_ids else None)
    elif chat_request.mode == "agent":
        chat_request.tools = resolve_tools(chat_request.agent_tools)
        chat_request.tool_choice = AGENT_TOOL_CHOICE
        chat_request.temperature = AGENT_TEMPERATURE
        system_content = build_system_content(mode=chat_request.mode, task_type=chat_request.task_type, custom_system=current_system_message, chunks=None)
    else:
        raise HTTPException(status_code=400, detail="Invalid mode")

    chat_request.messages = [ChatMessage(role="system", content=system_content), *history_messages, ChatMessage(role="user", content=current_user_message)]

    return chat_request, str(chat_id), str(message_id), prefixes


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


async def get_or_create_chat(db, session, chat_id, project_id, current_user, content_user, now) -> ObjectId:
    if chat_id:
        chat_object_id = await validate_existing_chat(db, session, chat_id, project_id, current_user)

        return chat_object_id

    title = generate_chat_title(content_user)
    chat_doc = ChatModel(
        project_id=project_id,
        title=title,
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


def extract_message(chat_request: MavanChatCompletionRequest, role: str) -> str | None:
    msg = next(
        (
            msg for msg in chat_request.messages
            if msg.role == role and isinstance(msg.content, str) and msg.content.strip()
        ),
        None
    )
    return msg.content.strip() if msg else None


def generate_chat_title(text: str, max_chars: int = 30) -> str:
    return text[:max_chars] + "..." if len(text) > max_chars else text


async def load_chat_history_for_llm(chat_id: str, parent_message_id: str, limit: int = 50) -> list[ChatMessage]:
    chain = await load_message_ancestor_chain(parent_message_id, limit)
    if not all(m.get("chat_id") == chat_id for m in chain):
        raise HTTPException(status_code=403, detail="Message chain contains messages from another chat")

    chain = list(reversed(chain))
    messages: list[ChatMessage] = []
    for doc in chain:
        if doc.get("content_user"):
            messages.append(ChatMessage(role="user", content=doc["content_user"]))
        if doc.get("content_assistant"):
            messages.append(ChatMessage(role="assistant", content=doc["content_assistant"]))
    return messages
