from datetime import datetime, timezone
from typing import Any

from bson import ObjectId

from app.database import get_db
from app.domain.schemas import DomainOut, SourceOut


def _oid(id: str) -> ObjectId:
    return ObjectId(id)


async def create_domain(name: str, description: str) -> DomainOut:
    doc = {
        "name": name, "description": description, "status": "init",
        "model_checkpoint": None, "train_cycle": 0, "created_at": datetime.now(timezone.utc),
    }
    res = await get_db().domains.insert_one(doc)
    return DomainOut(id=str(res.inserted_id), **{k: v for k, v in doc.items()})


async def get_domain(domain_id: str) -> dict | None:
    return await get_db().domains.find_one({"_id": _oid(domain_id)})


async def update_domain(domain_id: str, **fields) -> None:
    await get_db().domains.update_one({"_id": _oid(domain_id)}, {"$set": fields})


async def list_sources(domain_id: str) -> list[SourceOut]:
    out = []
    async for doc in get_db().source_registry.find({"domain_id": domain_id}).sort("url", 1):
        out.append(SourceOut(
            id=str(doc["_id"]), domain_id=doc["domain_id"], url=doc["url"],
            title=doc.get("title", ""), status=doc.get("status", "discovered"),
        ))
    return out


async def add_source(domain_id: str, url: str, title: str = "", status: str = "discovered") -> SourceOut:
    doc = {"domain_id": domain_id, "url": url, "title": title, "status": status,
           "created_at": datetime.now(timezone.utc)}
    res = await get_db().source_registry.insert_one(doc)
    return SourceOut(id=str(res.inserted_id), domain_id=domain_id, url=url, title=title, status=status)


async def upsert_sources(domain_id: str, urls: list[dict]) -> int:
    n = 0
    for item in urls:
        if await get_db().source_registry.find_one({"domain_id": domain_id, "url": item["url"]}):
            continue
        await add_source(domain_id, item["url"], item.get("title", ""), "discovered")
        n += 1
    return n


async def delete_source(domain_id: str, source_id: str) -> bool:
    res = await get_db().source_registry.delete_one({"_id": _oid(source_id), "domain_id": domain_id})
    return res.deleted_count > 0


async def approve_sources(domain_id: str, source_ids: list[str] | None = None) -> int:
    filt: dict[str, Any] = {"domain_id": domain_id}
    if source_ids:
        filt["_id"] = {"$in": [_oid(s) for s in source_ids]}
    else:
        filt["status"] = "discovered"
    res = await get_db().source_registry.update_many(filt, {"$set": {"status": "approved"}})
    return res.modified_count


async def get_approved_sources(domain_id: str) -> list[dict]:
    return await get_db().source_registry.find({"domain_id": domain_id, "status": "approved"}).to_list(500)


async def add_core_memory(domain_id: str, samples: list[dict]) -> None:
    if not samples:
        return
    now = datetime.now(timezone.utc)
    await get_db().core_memory.insert_many([
        {"domain_id": domain_id, "state": s["state"], "action": s["action"],
         "outcome_states": s["outcome_states"], "quality_avg": s.get("quality_avg", 0), "created_at": now}
        for s in samples
    ])


async def get_core_memory(domain_id: str) -> list[dict]:
    return await get_db().core_memory.find({"domain_id": domain_id}).to_list(5000)
