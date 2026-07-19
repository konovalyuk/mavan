from fastapi import APIRouter, Body, Depends

from app.models.auth_model import MavanUser
from app.models.domain_model import DomainCreate, DomainOut, PipelineRequest
from app.services.auth_service import get_user_from_token
from app.services import domain_conveyor

router = APIRouter()


@router.post("/api/v1/domains", response_model=DomainOut)
async def create_domain(body: DomainCreate = Body(...), current_user: MavanUser = Depends(get_user_from_token)):
    return await domain_conveyor.create_domain(body)


@router.get("/api/v1/domains/{domain_id}", response_model=DomainOut)
async def get_domain(domain_id: str, current_user: MavanUser = Depends(get_user_from_token)):
    return await domain_conveyor.get_domain(domain_id)


@router.post("/api/v1/domains/{domain_id}/pipeline/start")
async def start_conveyor(domain_id: str, body: PipelineRequest = Body(PipelineRequest()),
                         current_user: MavanUser = Depends(get_user_from_token)):
    return await domain_conveyor.start_conveyor(domain_id, provider=body.provider)


@router.post("/api/v1/domains/{domain_id}/pipeline/stop")
async def stop_conveyor(domain_id: str, current_user: MavanUser = Depends(get_user_from_token)):
    return await domain_conveyor.stop_conveyor(domain_id)
