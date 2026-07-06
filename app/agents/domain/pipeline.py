from app.agents.domain.extract import extract_transitions
from app.agents.domain.loop import run_discovery
from app.agents.domain.tools import fetch_url_text
from app.domain import store
from app.domain.settings import domain_settings
from app.domain_model.train import train_cycle
from app.quality.assessor import assess


class PipelineFailed(Exception):
    pass


async def run_domain_pipeline(domain_id: str, *, provider: str | None = None) -> dict:
    if not await store.get_domain(domain_id):
        raise ValueError("domain not found")

    for cycle in range(domain_settings.QUALITY_CYCLE_MAX):
        await store.update_domain(domain_id, status="collecting", train_cycle=cycle + 1)
        await run_discovery(domain_id, provider=provider)

        approved = await store.get_approved_sources(domain_id)
        if not approved:
            return {"status": "awaiting_approval", "cycle": cycle,
                    "message": "Approve sources then restart pipeline"}

        batch: list[dict] = []
        for src in approved:
            text = fetch_url_text(src["url"])
            if not text:
                continue
            qr = await assess(text, provider=provider)
            if qr["passed"]:
                batch.extend(await extract_transitions(text, qr["avg"], provider=provider))

        if not batch:
            continue

        await store.update_domain(domain_id, status="training")
        result = await train_cycle(domain_id, batch)

        if result == "ready":
            await store.add_core_memory(domain_id, batch)
            return {"status": "ready", "cycle": cycle, "samples": len(batch)}

        if result.startswith("insufficient"):
            await store.add_core_memory(domain_id, batch)
            continue

        await store.update_domain(domain_id, status="failed")
        raise PipelineFailed(result)

    await store.update_domain(domain_id, status="failed")
    raise PipelineFailed("max cycles reached without ready model")
