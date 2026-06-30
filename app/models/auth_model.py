from pydantic import BaseModel, Field


class MavanUser(BaseModel):
    username: str
    authorities: list[str] = Field(default_factory=lambda: ["user"])


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"