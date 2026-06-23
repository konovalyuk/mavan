from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import Dict, List, Any

from fastapi import HTTPException
from pymongo.errors import PyMongoError

import logging
import re

logger = logging.getLogger(__name__)

ZERO_WIDTH_CHARS_LIST = ["\u200B", "\u200C", "\u200D", "\uFEFF"]
_ZERO_WIDTH_CHARS_STR = "".join(ZERO_WIDTH_CHARS_LIST)
ZERO_WIDTH_CLASS = f"[{_ZERO_WIDTH_CHARS_STR}]"
INTER_WORD_SEP = ZERO_WIDTH_CLASS + "*" + r"\s*" + ZERO_WIDTH_CLASS + "*"


def build_space_insensitive_regex(user_text: str) -> str:
    """
    Build a regex that matches the user's words even if between them
    there are zero-width unicode chars (ZWNJ/ZWJ/FEFF) and/or normal whitespace.
    Returns a regex string (not compiled).
    """
    if not user_text:
        return ""
    zw_re = f"[{_ZERO_WIDTH_CHARS_STR}]"
    cleaned = re.sub(zw_re, " ", user_text)

    parts = re.split(r"\s+", cleaned.strip())
    parts = [p for p in parts if p]
    if not parts:
        return ""

    escaped_parts = [re.escape(p) for p in parts]
    return INTER_WORD_SEP.join(escaped_parts)


async def search_messages_by_text(db: AsyncIOMotorDatabase, regex: str, pattern: re.Pattern, search_in: List[str],
                                  username: str) -> List[Dict[str, Any]]:
    or_clauses = []
    if "content_user" in search_in:
        or_clauses.append({"content_user": {"$regex": regex, "$options": "i"}})
    if "content_assistant" in search_in:
        or_clauses.append({"content_assistant": {"$regex": regex, "$options": "i"}})

    if not or_clauses:
        return []

    message_filter: Dict[str, Any] = {"$or": or_clauses}
    if username:
        message_filter["created_by"] = username

    try:
        match_stage = {"$match": message_filter}

        group_stage = {
            "$group": {
                "_id": "$chat_id",
                "message": {
                    "$first": {
                        "_id": "$_id",
                        "chat_id": "$chat_id",
                        "content_user": "$content_user",
                        "content_assistant": "$content_assistant"
                    }
                }
            }
        }

        add_fields_stage = {
            "$addFields": {
                "chat_oid": {
                    "$convert": {
                        "input": "$message.chat_id",
                        "to": "objectId",
                        "onError": None,
                        "onNull": None
                    }
                }
            }
        }

        lookup_stage = {
            "$lookup": {
                "from": "chats",
                "localField": "chat_oid",
                "foreignField": "_id",
                "as": "chat"
            }
        }

        pipeline = [
            match_stage,
            group_stage,
            add_fields_stage,
            lookup_stage,
            {"$unwind": "$chat"},
            {"$sort": {"chat.updated_at": -1}}
        ]

        results = []
        async for doc in db.messages.aggregate(pipeline):
            message = doc["message"]
            chat = doc["chat"]

            for field in ["content_user", "content_assistant"]:
                if field in search_in and message.get(field):
                    content = message.get(field)
                    if not content:
                        continue
                    match = pattern.search(content)
                    if match:
                        snippet = extract_snippet_with_highlight(content, match.start(), match.end())
                        results.append({
                            "chat_id": str(chat.get("_id")),
                            "message_id": str(message.get("_id")),
                            "field": field,
                            "match_snippet": snippet,
                            "title": chat.get("title", ""),
                            "updated_at": chat.get("updated_at"),
                            **({"project_id": chat["project_id"]} if "project_id" in chat else {})
                        })
                        break
        return results
    except PyMongoError as e:
        logger.exception("MongoDB error in search_messages_by_text: %s", str(e))
        raise HTTPException(status_code=500, detail="Database error occurred")


async def search_chats_by_title(db: AsyncIOMotorDatabase, regex: str, pattern: re.Pattern, username: str,
                                exclude_ids: set[str]) -> List[Dict[str, Any]]:
    chat_filter: Dict[str, Any] = {
        "title": {"$regex": regex, "$options": "i"},
        "_id": {"$nin": [ObjectId(cid) for cid in exclude_ids]}
    }
    if username:
        chat_filter["created_by"] = username

    try:
        cursor = db.chats.find(chat_filter).sort("updated_at", -1)
        results = []
        async for chat in cursor:
            title = chat.get("title", "")
            if not title:
                continue
            match = pattern.search(title)
            if match:
                snippet = extract_snippet_with_highlight(title, match.start(), match.end())
                results.append({
                    "chat_id": str(chat.get("_id")),
                    "message_id": None,
                    "field": "title",
                    "match_snippet": snippet,
                    "title": chat.get("title", ""),
                    "updated_at": chat.get("updated_at"),
                    **({"project_id": chat["project_id"]} if "project_id" in chat else {})
                })
        return results
    except PyMongoError as e:
        logger.exception("MongoDB error in search_chats_by_title: %s", str(e))
        raise HTTPException(status_code=500, detail="Database error occurred")


def extract_snippet_with_highlight(text: str, match_start: int, match_end: int, pre=10, post=50) -> str:
    start = max(match_start - pre, 0)
    end = min(match_end + post, len(text))
    before = text[start:match_start]
    match_text = text[match_start:match_end]
    after = text[match_end:end]
    return f"{before}<mark>{match_text}</mark>{after}"
