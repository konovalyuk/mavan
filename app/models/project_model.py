from datetime import datetime
from fastapi import HTTPException
from typing import Optional, Any
from pydantic import BaseModel, field_validator

from app.models.mongo_model import MongoConverter, BaseMongoModel

class ProjectBaseModel(BaseModel):
    title: str | None = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str | None) -> str | None:
        if v is None:
            raise HTTPException(status_code=400, detail="Project title can't equal null")
        if str(v).strip() == "":
            raise HTTPException(status_code=400, detail="Project title can't be empty")
        if len(v.strip()) > 50:
            raise HTTPException(status_code=400, detail="Project title can't exceed 50 characters")
        return v.strip()


class ProjectModel(ProjectBaseModel):
    description: str | None = None
    instruction: str | None = None

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v.strip()) > 300:
            raise HTTPException(status_code=400, detail="Project description can't exceed 300 characters")
        return v.strip() if v else v


class ProjectInDB(ProjectModel, BaseMongoModel):
    pass

class ProjectBaseResponse(ProjectBaseModel, MongoConverter):
    created_at: datetime


class ProjectResponse(ProjectModel, MongoConverter):
    pass
