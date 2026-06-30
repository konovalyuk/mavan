from datetime import datetime
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class DocumentModel(BaseModel):
    """
    Model for documents from external systems (e.g., SSystem).
    Documents are already in text form and don't require file conversion.
    """
    type: Literal["document"]
    document_id: int
    subject: str
    author: str
    document_updated_at: datetime
    plain_text: str | None = None


class FileModel(BaseModel):
    """
    Model for uploaded files.
    Files may require conversion to extract text content.
    """
    type: Literal["file"]
    file_name: str
    file_type: str
    file_path: str | None = None
    file_size: int | None = None  # in bytes


AttachmentMeta = Annotated[Union[DocumentModel, FileModel], Field(discriminator="type")]


class AttachmentBaseModel(BaseModel):
    """
    Base model for all attachments (files and documents).
    Supports extensibility for future attachment sources.
    """
    state: str  # e.g., 'uploaded', 'deleted', 'available', 'not available', 'uploading', 'processing', 'ready'
    project_id: str | None = None
    chat_id: str | None = None
    message_ids: list[str] | None = None
    meta: AttachmentMeta
    created_at: datetime | None = None
    created_by: str | None = None
    updated_at: datetime | None = None
    updated_by: str | None = None


class FileAttachmentResponse(BaseModel):
    """Response model for file attachments"""
    id: str
    type: Literal["file"]
    file_name: str
    file_type: str


class DocumentAttachmentResponse(BaseModel):
    """Response model for document attachments"""
    id: str
    type: Literal["document"]
    document_id: int
    subject: str
    author: str
    document_updated_at: datetime
    project_id: str | None = None
    status: str
    updated_at: datetime | None = None
