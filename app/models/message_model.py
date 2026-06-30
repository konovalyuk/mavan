from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime
from app.models.mongo_model import MongoConverter


class MessageBase(BaseModel):
    attachment_ids: Optional[list[str]] = None  # Unified field for both files and documents
    content_user: str
    content_assistant: Optional[str] = None
    reaction: Optional[Literal["like", "dislike", "neutral"]] = None
    reaction_comment: Optional[str] = None
    created_at: datetime


class MessageCreate(MessageBase):
    chat_id: str
    children_count: int = 0
    node_path: str
    task_type: Optional[str]  # Changed from Literal to str to support any task type from MongoDB
    content_system: Optional[str]
    created_by: str
    updated_at: datetime | None = None
    updated_by: str | None = None


class MessageResponse(MessageBase, MongoConverter):
    sibling_position: int | None = None
    siblings_count: int | None = None
    parent_message_id: str | None = None


class MessageReactionResponse(MongoConverter):
    chat_id: str
    reaction: Optional[Literal["like", "dislike", "neutral"]] = None
    reaction_comment: Optional[str] = None


class ActivateChildRequest(BaseModel):
    sibling_position: int


class ActivateChildResponse(BaseModel):
    active_message_id: str
