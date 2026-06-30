from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from app.models.mongo_model import MongoConverter


class ChatBaseModel(BaseModel):
    project_id: str | None = None
    title: str


class ChatModel(ChatBaseModel):
    model: str
    active_message_id: str | None = None
    root_count: int = 0
    created_at: datetime
    created_by: str
    updated_at: datetime
    updated_by: str


class ChatUpdate(BaseModel):
    title: str


class ChatBaseResponse(ChatBaseModel, MongoConverter):
    updated_at: datetime


class ChatResponse(ChatModel, MongoConverter):
    pass
