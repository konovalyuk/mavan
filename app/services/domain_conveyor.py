import logging
import asyncio
from fastapi import HTTPException
from pymongo.errors import PyMongoError
from app.models.domain_model import DomainCreate, DomainOut, DomainDocument
from app.models.domain_model import DomainRunHandle
from app.database import get_db
from app.services.util_service import parse_object_id
from app.agents.runtime import SessionState, run_agent
from app.domain import store
from app.agents.domain.packet_agents import COORDINATOR, build_coordinator_execute, coordinator_agent, reset_packet

logger = logging.getLogger(__name__)

_RUNNING: dict[str, "DomainRunHandle"] = {}


def get_running(domain_id: str) -> DomainRunHandle | None:
    return _RUNNING.get(domain_id)


async def create_domain(domain: DomainCreate) -> DomainOut:
    try:
        if not domain.name or str(domain.name).strip() == "":
            raise HTTPException(status_code=400, detail="Domain name is required")
        domain_doc = DomainDocument(**domain.model_dump()).model_dump()
        result = await get_db().domains.insert_one(domain_doc)
        logger.info("Created new domain: %s", str(result.inserted_id))
        domain_doc["_id"] = result.inserted_id
        return DomainOut.from_mongo(domain_doc)
    except PyMongoError as e:
        logger.exception("MongoDB error while creating domain: %s", e)
        raise HTTPException(status_code=500, detail="Database error")
    except Exception as e:
        logger.exception("Unexpected error while creating domain: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


async def get_domain(domain_id: str) -> DomainOut:
    try:
        object_id = parse_object_id(domain_id)
        domain_dict = await get_db().domains.find_one({"_id": object_id})
        if not domain_dict:
            raise HTTPException(status_code=404, detail=f"Domain with id {domain_id} not found")
        return DomainOut.from_mongo(domain_dict)
    except Exception as e:
        logger.error("Failed to find domain (domain_id=%s): %s", domain_id, str(e), exc_info=True)
        raise


async def _run_until_stop(handle: DomainRunHandle) -> None:
    state: SessionState = handle.state
    while not handle.stop_event.is_set():
        try:
            reset_packet(state)
            execute = await build_coordinator_execute(state)
            user = (
                f"Train one atomic packet for domain_id={state.domain_id}.\n"
                f"Name: {state.domain_name}\nDescription: {state.domain_description}\n"
                "Flow: ResearchAgent → run_quality_team → ExtractAgent → TrainAgent. "
                "On failure use feedback and refine via ResearchAgent."
            )
            result = await run_agent(coordinator_agent(state), user, execute)
            state.metrics["coordinator_rounds"] = int(state.metrics.get("coordinator_rounds", 0)) + result.rounds
            state.packet_text = ""
            state.packet_urls = []
            await store.update_domain(state.domain_id, metrics=dict(state.metrics))
        except Exception:
            logger.exception("packet cycle failed domain=%s", handle.domain_id)
        if handle.stop_event.is_set():
            break
        await asyncio.sleep(0.5)


async def start_conveyor(domain_id: str, provider: str | None = None) -> dict:
    if domain_id in _RUNNING and not _RUNNING[domain_id].stop_event.is_set():
        return {"status": "running", "domain_id": domain_id, "message": "already running"}

    domain = await get_domain(domain_id)

    stop_event = asyncio.Event()
    state = SessionState(
        domain_id=domain_id,
        domain_name=domain.name,
        domain_description=domain.description,
        provider_override=provider,
    )
    handle = DomainRunHandle(domain_id, state, stop_event)
    await store.update_domain(domain_id, status="running", metrics=dict(state.metrics))
    _RUNNING[domain_id] = handle
    handle.task = asyncio.create_task(_run_until_stop(handle), name=f"domain-wf:{domain_id}")
    return {
        "status": "running",
        "domain_id": domain_id,
        "coordinator": COORDINATOR,
        "metrics": state.metrics,
    }


async def stop_conveyor(domain_id: str) -> dict:
    handle = _RUNNING.get(domain_id)
    if not handle:
        await get_domain(domain_id)
        await store.update_domain(domain_id, status="stopped")
        return {"status": "stopped", "domain_id": domain_id, "message": "was not running"}

    handle.stop_event.set()
    if handle.task:
        handle.task.cancel()
        await asyncio.gather(handle.task, return_exceptions=True)
    _RUNNING.pop(domain_id, None)
    await store.update_domain(domain_id, status="stopped", metrics=dict(handle.state.metrics))
    return {"status": "stopped", "domain_id": domain_id, "metrics": handle.state.metrics}
