from datetime import datetime, timezone

from app.database import get_db
from app.services.util_service import parse_object_id
from config import domain_settings


async def update_domain(domain_id: str, **fields) -> None:
    await get_db().domains.update_one({"_id": parse_object_id(domain_id)}, {"$set": fields})


async def add_replay(domain_id: str, samples: list[dict]) -> None:
    """Append compressed S-A-O snapshots; trim to CORE_MEMORY_MAX_SAMPLES (ring)."""
    if not samples:
        return
    now = datetime.now(timezone.utc)
    col = get_db().core_memory
    await col.insert_many([
        {
            "domain_id": domain_id,
            "state": s["state"],
            "action": s["action"],
            "outcome_states": s["outcome_states"],
            "quality_avg": s.get("quality_avg", 0),
            "created_at": now,
        }
        for s in samples
    ])
    cap = domain_settings.CORE_MEMORY_MAX_SAMPLES
    total = await col.count_documents({"domain_id": domain_id})
    if total <= cap:
        return
    overflow = total - cap
    old = await col.find({"domain_id": domain_id}).sort("created_at", 1).limit(overflow).to_list(overflow)
    ids = [d["_id"] for d in old]
    if ids:
        await col.delete_many({"_id": {"$in": ids}})


async def get_core_memory(domain_id: str, limit: int | None = None) -> list[dict]:
    """Capped S-A-O replay buffer (newest first, then chronological for training)."""
    cap = limit or domain_settings.CORE_MEMORY_MAX_SAMPLES
    docs = await get_db().core_memory.find({"domain_id": domain_id}).sort("created_at", -1).to_list(cap)
    docs.reverse()
    return docs


# Back-compat alias used by train / forecast
add_core_memory = add_replay
