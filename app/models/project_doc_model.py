from pydantic import BaseModel
from datetime import datetime


class DocumentBaseModel(BaseModel):
    document_id: int
    subject: str
    author: str
    document_updated_at: datetime


class DocumentModel(DocumentBaseModel):
    project_id: str
    status: str
