import logging
from typing import Optional, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from app.models.auth_model import MavanUser
from config import auth_settings
from app.database import get_db
from datetime import datetime, timedelta, timezone
import bcrypt
import jwt

logger = logging.getLogger(__name__)

auth_scheme = HTTPBearer(auto_error=False)


def _dev_user() -> MavanUser:
    return MavanUser(
        username=auth_settings.DEV_USERNAME,
        authorities=["user"],
    )


def decode_access_token(credentials: HTTPAuthorizationCredentials) -> MavanUser:
    try:
        payload = jwt.decode(
            credentials.credentials,
            auth_settings.JWT_SECRET_KEY,
            algorithms=[auth_settings.JWT_ALGORITHM],
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired", headers={"WWW-Authenticate": "Bearer"})
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token", headers={"WWW-Authenticate": "Bearer"})

    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload", headers={"WWW-Authenticate": "Bearer"})

    authorities = payload.get("authorities", ["user"])
    if not isinstance(authorities, list):
        authorities = ["user"]

    return MavanUser(username=username, authorities=authorities)


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
    return decode_access_token(credentials)


def require_roles(*required_roles: str):
    async def dependency(user: MavanUser = Depends(get_user_from_token)) -> MavanUser:
        if not any(role in user.authorities for role in required_roles):
            logger.warning(f"Access denied for user {user.username} — requires one of: {required_roles}")
            raise HTTPException(status_code=403, detail=f"Required roles: {required_roles}")
        return user

    return dependency


async def authenticate_user(username: str, password: str) -> MavanUser:
    doc = await get_db().users.find_one({"username": username, "is_active": True})
    if not doc or not verify_password(password, doc["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    return MavanUser(username=doc["username"], authorities=doc.get("authorities", ["user"]))


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(
        plain.encode("utf-8"),
        hashed.encode("utf-8"),
    )


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(
        plain.encode("utf-8"),
        bcrypt.gensalt(),
    ).decode("utf-8")


def create_access_token(*, username: str, authorities: list[str] | None = None) -> str:
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=auth_settings.JWT_EXPIRE_MINUTES)
    payload: dict[str, Any] = {
        "sub": username,
        "authorities": authorities or ["user"],
        "iat": now,
        "exp": expire,
    }
    return jwt.encode(
        payload,
        auth_settings.JWT_SECRET_KEY,
        algorithm=auth_settings.JWT_ALGORITHM,
    )