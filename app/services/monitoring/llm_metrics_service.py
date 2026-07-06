import time
from datetime import datetime, timezone

from app.database import get_db
from app.models.llm_call_model import LLMCallRecord


async def record_llm_call(record: LLMCallRecord) -> None:
    doc = {
        **record.__dict__,
        "created_at": datetime.now(timezone.utc),
    }
    await get_db().llm_calls.insert_one(doc)