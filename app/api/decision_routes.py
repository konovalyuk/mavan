from fastapi import APIRouter, Body, Depends, HTTPException

from app.domain.schemas import DecisionRequest, ForecastRequest
from app.models.auth_model import MavanUser
from app.services.auth_service import get_user_from_token
from app.services import decision_service

router = APIRouter()


@router.post("/api/v1/decisions/forecast")
async def forecast(body: ForecastRequest = Body(...), _user: MavanUser = Depends(get_user_from_token)):
    try:
        return await decision_service.forecast(body)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/api/v1/decisions/recommend")
async def recommend(body: DecisionRequest = Body(...), _user: MavanUser = Depends(get_user_from_token)):
    try:
        return await decision_service.recommend(body)
    except ValueError as e:
        raise HTTPException(400, str(e))
