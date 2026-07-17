from pydantic import BaseModel, field_validator
from datetime import datetime
from typing import Literal, Optional
from fastapi import HTTPException

from app.llm.chat.schemas import ChatCompletionRequest, ChatCompletionResponse
from app.models.mongo_model import MongoConverter


class MavanChatCompletionRequest(ChatCompletionRequest):
    """OpenAI chat fields + Mavan product fields (not part of OpenAI schema)."""

    project_id: Optional[str] = None
    chat_id: Optional[str] = None
    parent_message_id: Optional[str] = None
    # Product task preset: translate | summarize | transcribe | qa | …
    # Maps to TASK_INSTRUCTIONS. Not an OpenAI field — use system.content for freeform instructions.
    task_type: Optional[str] = None
    provider_name: Optional[str] = None
    rag_top_k: int = 5
    attachment_ids: list[str] | None = None
    mode: Literal["ask", "agent"] = "ask"
    agent_tools: str | list[str] | None = None
    history_limit: int = 10
    sources: list[dict] = []

    @field_validator("chat_id")
    @classmethod
    def validate_chat_id(cls, v):
        if v is None or str(v).strip() == "":
            raise HTTPException(status_code=400, detail="Chat id can't be empty string or null")
        return v

    @field_validator("project_id")
    @classmethod
    def validate_project_id(cls, v):
        if v is None or str(v).strip() == "":
            raise HTTPException(status_code=400, detail="Project id can't be empty string or null")
        return v

    @field_validator("stream")
    @classmethod
    def validate_stream(cls, v):
        if v is None:
            raise HTTPException(
                status_code=400,
                detail="Stream parameter cannot be null. Use true for streaming or false for non-streaming response.",
            )
        return v


class MavanChatCompletionResponse(ChatCompletionResponse):
    mode: str | None = None
    sources: list[dict] | None = None
    chat_id: str
    message_id: str


class ChatBaseModel(BaseModel):
    project_id: str | None = None
    title: str


class ChatModel(ChatBaseModel):
    model: str  | None = None
    attachment_ids: list[str] = []
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
