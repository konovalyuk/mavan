import asyncio
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from app.models.mongo_model import MongoConverter
from app.agents.runtime import SessionState


class PipelineRequest(BaseModel):
    provider: str | None = None


class DomainCreate(BaseModel):
    name: str
    description: str = ""


class DomainDocument(DomainCreate):
    status: str = "stopped"
    model_checkpoint: str | None = None
    train_steps: int = 0
    train_cycle: int = 0
    metrics: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DomainOut(DomainDocument, MongoConverter):
    pass


class DomainRunHandle:
    def __init__(self, domain_id: str, state: SessionState, stop_event: asyncio.Event):
        self.domain_id = domain_id
        self.state = state
        self.stop_event = stop_event
        self.task: asyncio.Task | None = None