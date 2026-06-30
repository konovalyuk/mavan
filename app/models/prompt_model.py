from typing import Optional
from bson import ObjectId
from pydantic import BaseModel, Field, field_validator
from pydantic_core import PydanticCustomError

from app.models.mongo_model import BaseMongoModel


class PromptBase(BaseModel):
    label: str
    icon: Optional[str] = ""
    content: str


class PromptCreate(PromptBase):
    @field_validator("label")
    @classmethod
    def validate_label(cls, v):
        if not v or not v.strip():
            raise PydanticCustomError("label_error", "Label can't be null or empty")
        return v

    @field_validator("content")
    @classmethod
    def validate_content(cls, v):
        if not v or not v.strip():
            raise PydanticCustomError("content_error", "Content can't be null or empty")
        return v


class PromptInDB(PromptBase, BaseMongoModel):
    pass


class PromptPreview(BaseModel):
    id: ObjectId = Field(alias="_id")
    label: str
    icon: Optional[str] = ""
    content: str

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class PromptResponse(BaseModel):
    status: str = "success"
    id: str
