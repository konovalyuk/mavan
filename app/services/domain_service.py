from app.domain import store
from app.domain.schemas import DomainCreate, DomainOut, SourceCreate, SourceOut


async def create_domain(body: DomainCreate) -> DomainOut:
    return await store.create_domain(body.name, body.description)


async def get_domain(domain_id: str) -> DomainOut | None:
    doc = await store.get_domain(domain_id)
    if not doc:
        return None
    return DomainOut(
        id=str(doc["_id"]), name=doc["name"], description=doc.get("description", ""),
        status=doc.get("status", "init"), model_checkpoint=doc.get("model_checkpoint"),
        train_cycle=doc.get("train_cycle", 0), created_at=doc["created_at"],
    )


async def list_sources(domain_id: str) -> list[SourceOut]:
    return await store.list_sources(domain_id)


async def add_source(domain_id: str, body: SourceCreate) -> SourceOut:
    return await store.add_source(domain_id, body.url, body.title, "discovered")


async def delete_source(domain_id: str, source_id: str) -> bool:
    return await store.delete_source(domain_id, source_id)


async def approve_sources(domain_id: str, source_ids: list[str] | None = None) -> int:
    return await store.approve_sources(domain_id, source_ids)


async def start_pipeline(domain_id: str, provider: str | None = None) -> dict:
    from app.agents.domain.pipeline import run_domain_pipeline
    doc = await store.get_domain(domain_id)
    if not doc:
        raise ValueError("domain not found")
    await store.update_domain(domain_id, status="collecting")
    result = await run_domain_pipeline(domain_id, provider=provider)
    return result
