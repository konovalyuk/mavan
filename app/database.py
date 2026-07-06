import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT

from config import mongo_settings

logger = logging.getLogger(__name__)

_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(mongo_settings.URI)
    return _client


def get_db() -> AsyncIOMotorDatabase:
    global _db
    if _db is None:
        client = get_client()
        _db = client[mongo_settings.DATABASE]
    return _db


async def create_indexes():
    """
    Creates necessary indexes for collections.
    Should be called on application startup.
    """
    db = get_db()

    try:
        await db.messages.create_indexes([
            IndexModel([("created_by", ASCENDING)]),
            IndexModel([("chat_id", ASCENDING)]),
            IndexModel([("content_user", TEXT), ("content_assistant", TEXT)]),
            IndexModel([("chat_id", ASCENDING), ("created_at", DESCENDING)], name="chat_created_desc"),
            IndexModel([("parent_message_id", ASCENDING)], name="parent_lookup"),
            IndexModel([("parent_message_id", ASCENDING), ("created_at", ASCENDING)], name="parent_created"),
            IndexModel([("_id", ASCENDING), ("chat_id", ASCENDING)], name="id_chat"),
            # PERF: Compound indexes for optimized chat history and last-message queries
            IndexModel([("chat_id", ASCENDING), ("created_at", DESCENDING), ("role", ASCENDING)], name="idx_chat_history"),
            IndexModel([("chat_id", ASCENDING), ("role", ASCENDING), ("created_at", DESCENDING)], name="idx_last_user_message"),
            IndexModel([("chat_id", ASCENDING), ("node_path", ASCENDING)], name="idx_chat_nodepath"),
            IndexModel([("chat_id", ASCENDING), ("parent_message_id", ASCENDING)], name="idx_chat_parent"),
            IndexModel([("chat_id", ASCENDING), ("created_by", ASCENDING), ("node_path", ASCENDING)], name="idx_chat_user_nodepath"),
        ])
        logger.info("Indexes created on messages (including compound indexes)")

        await db.chats.create_indexes([
            IndexModel([("created_by", ASCENDING), ("updated_at", DESCENDING)]),
            IndexModel([("title", TEXT)]),
            IndexModel([("created_by", ASCENDING)], name="chat_owner"),
        ])
        logger.info("Compound index created on chats (created_by, updated_at)")

        # Create indexes for prompts collection
        await db.prompts.create_indexes([
            IndexModel([("name", ASCENDING)]),
            IndexModel([("label", ASCENDING)])
        ])
        logger.info("Indexes created on prompts collection (name, label)")

        # Create indexes for projects collection
        await db.projects.create_indexes([
            IndexModel([("created_by", ASCENDING), ("created_at", DESCENDING)]),
            IndexModel([("title", TEXT), ("description", TEXT), ("instruction", TEXT)]),
            # PERF: Compound index for project lookups by ID + owner
            IndexModel([("_id", ASCENDING), ("created_by", ASCENDING)], name="idx_project_user"),
        ])
        logger.info("Indexes created on projects (including compound indexes)")

        # Create indexes for attachments collection
        await db.attachments.create_indexes([
            IndexModel([("created_by", ASCENDING), ("state", ASCENDING)]),
            IndexModel([("chat_id", ASCENDING)]),
            IndexModel([("message_ids", ASCENDING)]),
            IndexModel([("meta.file_type", ASCENDING)]),
            # PERF: Compound index for attachment lookups by ID + owner + state
            IndexModel([("_id", ASCENDING), ("created_by", ASCENDING), ("state", ASCENDING)], name="idx_attachment_user_state"),
        ])
        logger.info("Indexes created on attachments (including compound indexes)")

        # Create indexes for llm calls collection
        await db.llm_calls.create_indexes([
            IndexModel([("created_at", DESCENDING)]),
            IndexModel([("capability", ASCENDING), ("created_at", DESCENDING)]),
        ])
        logger.info("Indexes created on llm_calls")

    except Exception as e:
        logger.exception("Failed to create MongoDB indexes: %s", str(e))
