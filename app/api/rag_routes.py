from fastapi import APIRouter, Body, Depends
from pydantic import BaseModel

from app.models.auth_model import MavanUser
from app.services.auth_service import get_user_from_token
from app.services.rag_service import answer_question

router = APIRouter()


class RagQueryRequest(BaseModel):
    question: str
    provider: str | None = None
    top_k: int = 5


class RagQueryResponse(BaseModel):
    answer: str
    sources: list[dict]


@router.post("/api/v1/rag/query")
async def rag_query(
        body: RagQueryRequest = Body(...),
        current_user: MavanUser = Depends(get_user_from_token),
):
    result = await answer_question(body.question, provider=body.provider, top_k=body.top_k)
    return RagQueryResponse(
        answer=result.text,
        sources=[
            {"text": c.text, "score": c.score, "source": c.source}
            for c in result.sources
        ],
    )
