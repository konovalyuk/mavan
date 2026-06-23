from datetime import datetime, timedelta, timezone
from typing import Any
import bcrypt
import jwt

from config import auth_settings
from app.models.auth_model import MavanUser


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


def decode_access_token(token: str) -> MavanUser:
    from app.models.auth_model import MavanUser  # локальный import против циклов

    try:
        payload = jwt.decode(
            token,
            auth_settings.JWT_SECRET_KEY,
            algorithms=[auth_settings.JWT_ALGORITHM],
        )
    except jwt.ExpiredSignatureError:
        raise ValueError("Token expired")
    except jwt.PyJWTError:
        raise ValueError("Invalid token")

    username = payload.get("sub")
    if not username:
        raise ValueError("Invalid token payload")

    authorities = payload.get("authorities", ["user"])
    if not isinstance(authorities, list):
        authorities = ["user"]

    return MavanUser(username=username, authorities=authorities)
