from fastapi import APIRouter, status

from app.models.auth_model import LoginRequest, TokenResponse
from app.services.auth_service import authenticate_user, create_access_token

router = APIRouter(tags=["auth"])


@router.post(
    "/api/v1/auth/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Login and get JWT access token",
)
async def login(body: LoginRequest) -> TokenResponse:
    user = await authenticate_user(body.username, body.password)
    token = create_access_token(username=user.username, authorities=user.authorities)
    return TokenResponse(access_token=token)
