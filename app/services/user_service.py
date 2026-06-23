from app.core.security import verify_password
from app.database import get_db
from app.models.auth_model import MavanUser
from fastapi import HTTPException, status


async def authenticate_user(username: str, password: str) -> MavanUser:
    doc = await get_db().users.find_one({"username": username, "is_active": True})
    if not doc or not verify_password(password, doc["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    return MavanUser(
        username=doc["username"],
        authorities=doc.get("authorities", ["user"]),
    )