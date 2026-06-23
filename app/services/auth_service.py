import logging
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from app.core.security import decode_access_token
from app.models.auth_model import MavanUser
from config import auth_settings

logger = logging.getLogger(__name__)

auth_scheme = HTTPBearer(auto_error=False)


def _dev_user() -> MavanUser:
    return MavanUser(
        username=auth_settings.DEV_USERNAME,
        authorities=["user"],
    )


def _user_from_jwt(credentials: HTTPAuthorizationCredentials) -> MavanUser:
    try:
        return decode_access_token(credentials.credentials)
    except ValueError as e:
        logger.warning("JWT validation failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_user_from_token(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(auth_scheme),
) -> MavanUser:
    if auth_settings.MODE == "dev":
        return _dev_user()
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _user_from_jwt(credentials)


def require_roles(*required_roles: str):
    async def dependency(user: MavanUser = Depends(get_user_from_token)) -> MavanUser:
        if not any(role in user.authorities for role in required_roles):
            logger.warning(f"Access denied for user {user.username} — requires one of: {required_roles}")
            raise HTTPException(status_code=403, detail=f"Required roles: {required_roles}")
        return user

    return dependency
