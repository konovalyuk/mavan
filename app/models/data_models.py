from typing import Optional

from fastapi import HTTPException
from pydantic import field_validator

from app.llm.chat.schemas import ChatMessage, ChatCompletionRequest


class MavanChatCompletionRequest(ChatCompletionRequest):
    """API request: OpenAI fields + mavan business fields."""

    project_id: Optional[str] = None
    chat_id: Optional[str] = None
    parent_message_id: Optional[str] = None
    task_type: Optional[str] = None
    provider: Optional[str] = None

    def to_provider_request(self) -> ChatCompletionRequest:
        return ChatCompletionRequest.model_validate(
            self.model_dump(include=set(ChatCompletionRequest.model_fields.keys()))
        )

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
