from fastapi import APIRouter, Body, Depends
from pydantic import BaseModel

from app.models.auth_model import MavanUser
from app.services.agent_service import run_agent
from app.services.auth_service import get_user_from_token

router = APIRouter()


class AgentRequest(BaseModel):
    question: str
    provider: str | None = None


class AgentResponse(BaseModel):
    answer: str
    agent: str
    sources: list[dict]


@router.post("/api/v1/agents/rag")
async def rag_agent(
        body: AgentRequest = Body(...),
        current_user: MavanUser = Depends(get_user_from_token),
):
    result = await run_agent(body.question, provider=body.provider)
    return AgentResponse(**result)
