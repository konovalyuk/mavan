from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel

from app.domain.schemas import DomainCreate, DomainOut, SourceCreate, SourceOut
from app.models.auth_model import MavanUser
from app.services.auth_service import get_user_from_token
from app.services import domain_service

router = APIRouter()


class ApproveRequest(BaseModel):
    source_ids: list[str] | None = None


class PipelineRequest(BaseModel):
    provider: str | None = None


@router.post("/api/v1/domains", response_model=DomainOut)
async def create_domain(body: DomainCreate = Body(...), _user: MavanUser = Depends(get_user_from_token)):
    return await domain_service.create_domain(body)


@router.get("/api/v1/domains/{domain_id}", response_model=DomainOut)
async def get_domain(domain_id: str, _user: MavanUser = Depends(get_user_from_token)):
    domain = await domain_service.get_domain(domain_id)
    if not domain:
        raise HTTPException(404, "domain not found")
    return domain


@router.get("/api/v1/domains/{domain_id}/sources", response_model=list[SourceOut])
async def list_sources(domain_id: str, _user: MavanUser = Depends(get_user_from_token)):
    return await domain_service.list_sources(domain_id)


@router.post("/api/v1/domains/{domain_id}/sources", response_model=SourceOut)
async def add_source(domain_id: str, body: SourceCreate = Body(...), _user: MavanUser = Depends(get_user_from_token)):
    return await domain_service.add_source(domain_id, body)


@router.delete("/api/v1/domains/{domain_id}/sources/{source_id}")
async def delete_source(domain_id: str, source_id: str, _user: MavanUser = Depends(get_user_from_token)):
    if not await domain_service.delete_source(domain_id, source_id):
        raise HTTPException(404, "source not found")
    return {"deleted": True}


@router.post("/api/v1/domains/{domain_id}/sources/approve")
async def approve_sources(domain_id: str, body: ApproveRequest = Body(ApproveRequest()),
                          _user: MavanUser = Depends(get_user_from_token)):
    n = await domain_service.approve_sources(domain_id, body.source_ids)
    return {"approved": n}


@router.post("/api/v1/domains/{domain_id}/pipeline/start")
async def start_pipeline(domain_id: str, body: PipelineRequest = Body(PipelineRequest()),
                         _user: MavanUser = Depends(get_user_from_token)):
    try:
        return await domain_service.start_pipeline(domain_id, provider=body.provider)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))
