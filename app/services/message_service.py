from bson import ObjectId
from typing import Literal
from datetime import datetime, timezone
from pymongo import DESCENDING, ReturnDocument
from fastapi import HTTPException, status
from app.database import get_db
from app.models.message_model import MessageReactionResponse, MessageResponse, ActivateChildResponse
from app.services.util_service import parse_object_id
import logging

logger = logging.getLogger(__name__)


async def load_chat_messages(chat_id: str, username: str, limit: int, parent_message_id: str | None = None) -> list[
    MessageResponse]:
    """
    Returns messages for chat based on active branch.
    - If parent_message_id is provided, builds chain from it.
    - Otherwise uses chat.active_message_id.
    - Falls back to legacy linear history if no active_message_id exists.
    """
    chat_dict = await load_chat_with_access_check(chat_id, username, projection={"created_by": 1, "active_message_id": 1, "root_count": 1})
    tail_message_id = parent_message_id or chat_dict.get("active_message_id")

    messages = await load_message_ancestor_chain(tail_message_id=tail_message_id, limit=limit)
    if not all(m.get("chat_id") == chat_id for m in messages):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Message chain contains messages from another chat")

    root_count = chat_dict.get("root_count", 0)

    children_map = {
        str(m["_id"]): m.get("children_count", 0)
        for m in messages
    }
    response: list[MessageResponse] = []

    if len(messages) > limit:
        messages = messages[:limit]
    for m in messages:
        node_path = m["node_path"]
        sibling_position = int(node_path.split(".")[-1])

        parent_id = m.get("parent_message_id")
        if parent_id:
            siblings_count = children_map.get(str(parent_id), 0)
        else:
            siblings_count = root_count

        m["sibling_position"] = sibling_position
        m["siblings_count"] = siblings_count
        m["parent_message_id"] = parent_id
        response.append(MessageResponse.from_mongo(m))
    return response


async def add_reaction_to_message(chat_id: str, message_id: str, reaction: Literal["like", "dislike", "neutral"] | None,
                                  reaction_comment: str,
                                  username: str) -> MessageReactionResponse:
    """
    Add or update reaction + comment on a message with the following rules:
      - dislike requires reaction_comment
      - like may be without comment
      - if message currently has reaction == "like" then client may send only reaction_comment (reaction may be None)
      - if message currently has reaction_comment (non-empty), then changing reaction is prohibited (400)
      - reaction == "neutral" clears reaction (or sets to neutral) only if:
            * request does NOT include reaction_comment, AND
            * message currently has NO reaction_comment
    """
    try:
        message_object_id = parse_object_id(message_id)
        message_filter = {"_id": message_object_id, "chat_id": chat_id}

        await validate_reaction_and_comment(chat_id, message_filter, reaction, reaction_comment, username)

        update_fields = {}

        if reaction is not None:
            update_fields["reaction"] = reaction
        if reaction_comment is not None:
            update_fields["reaction_comment"] = reaction_comment.strip()
        if not update_fields:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Nothing to update (no reaction or reaction_comment provided).")

        update_fields["updated_at"] = datetime.now(timezone.utc)
        update_fields["updated_by"] = username

        message_dict = await get_db().messages.find_one_and_update(
            message_filter,
            {"$set": update_fields},
            return_document=ReturnDocument.AFTER
        )
        if not message_dict:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found")

        return MessageReactionResponse.from_mongo(message_dict)
    except HTTPException as e:
        logger.error("Failed to add reaction to message %s in chat %s: %s", message_id, chat_id, str(e), exc_info=True)
        raise
    except Exception as e:
        logger.exception("Failed to add reaction to message %s in chat %s: %s", message_id, chat_id, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Internal Server Error while adding reaction")


async def validate_reaction_and_comment(chat_id: str, message_filter, reaction: str, reaction_comment: str,
                                        username: str):
    # validate chat exists and ownership
    chat_object_id = parse_object_id(chat_id)
    chat_filter = {"_id": chat_object_id}
    chat_dict = await get_db().chats.find_one(chat_filter, projection={"_id": 0, "created_by": 1})
    if not chat_dict:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
    if chat_dict["created_by"] != username:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this chat")

    # validate message
    message_doc = await get_db().messages.find_one(message_filter)
    if not message_doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found")

    current_reaction = message_doc.get("reaction")
    current_comment = message_doc.get("reaction_comment")  # may be None or empty

    # rule: dislike requires a comment in the request
    if reaction == "dislike" and (not reaction_comment or not reaction_comment.strip()):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Dislike reaction requires a non-empty reaction_comment")

    # neutral must be sent without a reaction_comment
    if reaction == "neutral" and reaction_comment:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Neutral reaction must be sent without reaction_comment")

    if reaction is None:
        # client sent only comment (or nothing). If no comment and no reaction -> nothing to change
        if not reaction_comment:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Must provide 'reaction' or 'reaction_comment'.")
        # if current reaction is not 'like', disallow comment-only update
        if current_reaction != "like":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Can only add comment without reaction when an existing 'like' is present.")

    if reaction == "neutral":
        if current_comment and str(current_comment).strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Cannot set neutral when a reaction_comment already exists.")

    # rule: if message already has reaction_comment (non-empty), then reaction cannot be changed
    if current_comment and current_comment.strip():
        if reaction is not None and reaction != current_reaction:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Reaction cannot be changed after a reaction_comment was set.")


async def load_message_ancestor_chain(tail_message_id: str, limit: int) -> list:
    """ Loads message chain starting from tail_message_id using $graphLookup to follow parent-child links."""
    if limit is None or limit < 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Limit must be greater or equal than 0")
    tail_object_id = parse_object_id(tail_message_id)

    pipeline = [
        {"$match": {"_id": tail_object_id}},
        {"$project": {"_id": 1, "parent_message_id": 1, "created_at": 1}},
        {
            "$graphLookup": {
                "from": "messages",
                "startWith": "$_id",
                "connectFromField": "parent_message_id",
                "connectToField": "_id",
                "as": "chain",
                "depthField": "depth",
                "maxDepth": limit if limit else None
            }
        },
        {"$unwind": "$chain"},
        {"$replaceRoot": {"newRoot": "$chain"}},
        {"$sort": {"depth": 1}},
        {"$limit": limit + 1}
    ]

    messages = await get_db().messages.aggregate(pipeline).to_list()
    if not messages:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The message wasn’t found")

    return messages


async def activate_message_branch(chat_id: str, sibling_position: int, username: str,
                                  parent_message_id: str | None = None) -> ActivateChildResponse:
    if sibling_position < 1:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sibling position can not be less than 1")

    chat = await load_chat_with_access_check(chat_id, username, projection={"root_count": 1, "created_by": 1})
    if not parent_message_id:
        if sibling_position > chat.get("root_count", 0):
            raise HTTPException(status_code=400, detail="Sibling position exceeds maximum value for root messages")
        node_path = str(sibling_position)

        sibling_message = await get_db().messages.find_one(
            {"chat_id": chat_id, "parent_message_id": None, "node_path": node_path, "created_by": username},
            projection={"_id": 1})
        if not sibling_message:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail="Root message not found for the given sibling position")
        sibling_message_id = sibling_message["_id"]
    else:
        parent_message_oid = parse_object_id(parent_message_id)
        parent_message = await load_message_with_access_check(parent_message_oid, chat_id, username,
                                                              projection={"node_path": 1, "children_count": 1,
                                                                          "chat_id": 1, "created_by": 1})
        if sibling_position > parent_message.get("children_count", 0):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Sibling position exceeds maximum value for child messages")
        node_path = f"{parent_message['node_path']}.{sibling_position}"

        sibling_message = await get_db().messages.find_one(
            {"chat_id": chat_id, "parent_message_id": parent_message_oid, "node_path": node_path,
             "created_by": username}, projection={"_id": 1})
        if not sibling_message:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail="Child message not found for the given sibling position")
        sibling_message_id = sibling_message["_id"]

    leaf_id = await find_deepest_leaf(sibling_message_id)

    result = await get_db().chats.update_one({"_id": parse_object_id(chat_id)},
                                             {"$set": {"active_message_id": leaf_id}})
    if result.matched_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat has been not found during activation")
    return ActivateChildResponse(active_message_id=leaf_id)


async def find_deepest_leaf(start_message_id: ObjectId) -> str:
    """ Finds the deepest leaf message ID in the branch starting from start_message_id using $graphLookup."""
    pipeline = [
        {"$match": {"_id": start_message_id}},
        {"$project": {"_id": 1, "parent_message_id": 1, "created_at": 1}},
        {
            "$graphLookup": {
                "from": "messages",
                "startWith": "$_id",
                "connectFromField": "_id",
                "connectToField": "parent_message_id",
                "as": "nodes",
                "depthField": "depth"
            }
        },
        {"$unwind": "$nodes"},
        {"$sort": {"nodes.depth": -1, "nodes.created_at": -1}},
        {"$limit": 1},
        {"$replaceRoot": {"newRoot": "$nodes"}},
        {"$project": {"_id": 1, "parent_message_id": 1, "created_at": 1}}
    ]
    messages = await get_db().messages.aggregate(pipeline).to_list()

    return str(messages[0]["_id"]) if messages else str(start_message_id)


async def load_message_with_access_check(message_id: ObjectId, chat_id: str, username: str,
                                         projection: dict | None = None):
    """ Loads message and checks that it belongs to the chat and was created by the user."""
    projection = projection or {"chat_id": 1, "created_by": 1}
    message = await get_db().messages.find_one({"_id": message_id}, projection=projection)
    if not message:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The message wasn’t found")
    if message["chat_id"] != chat_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Message does not belong to the specified chat")
    if message["created_by"] != username:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don’t have access to this message")
    return message


async def load_chat_with_access_check(chat_id: str, username: str, projection: dict | None = None):
    """ Loads chat and checks that it was created by the user."""
    projection = projection or {"created_by": 1}
    chat = await get_db().chats.find_one({"_id": parse_object_id(chat_id)}, projection=projection)
    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The chat wasn’t found")
    if chat["created_by"] != username:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don’t have access to this chat")
    return chat
