from typing import Any, TypeVar, Type
from pydantic import BaseModel, Field
from bson import ObjectId
from datetime import datetime

T = TypeVar("T", bound="MongoConverter")


class MongoConverter(BaseModel):
    id: str

    @classmethod
    def from_mongo(cls: Type[T], mongo_dict: dict) -> T:
        """
        Universal method for converting dict from MongoDB to Pydantic model.
        - _id -> id (str)
        - saves all other fields automatically
        """

        def convert(obj: Any) -> Any:
            if isinstance(obj, dict):
                obj = obj.copy()
                if "_id" in obj:
                    obj["id"] = str(obj.pop("_id"))
                for k, v in obj.items():
                    obj[k] = convert(v)
                return obj
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            elif isinstance(obj, ObjectId):
                return str(obj)
            else:
                return obj

        data = convert(mongo_dict)
        return cls(**data)


class BaseMongoModel(BaseModel):
    id: ObjectId = Field(alias="_id")
    created_at: datetime
    created_by: str
    updated_at: datetime
    updated_by: str

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
