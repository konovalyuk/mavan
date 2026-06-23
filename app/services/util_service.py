import logging
from bson import ObjectId, errors as bson_errors
from fastapi import HTTPException

logger = logging.getLogger(__name__)

def parse_object_id(id_str: str) -> ObjectId | None:
    if not id_str:
        return None
    try:
        return ObjectId(id_str)
    except bson_errors.InvalidId:
        logger.error("Invalid id format: %s.", id_str)
        raise HTTPException(status_code=400, detail="Invalid ID format")
