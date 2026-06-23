import logging
from datetime import timezone
from typing import List

from fastapi import status
from pymongo import DESCENDING, ReturnDocument
from pymongo.errors import PyMongoError, ConfigurationError

from app.database import get_db
from app.models.project_model import *
from app.services.util_service import parse_object_id

logger = logging.getLogger(__name__)


async def find_all_projects(username: str, limit: int, before: str | None) -> List[ProjectBaseResponse]:
    try:
        query = {"created_by": username}

        if before:
            try:
                before_dt = datetime.fromisoformat(before.replace("Z", "+00:00"))
                query["created_at"] = {"$lt": before_dt}
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid 'before' datetime format")

        cursor = get_db().projects.find(
            query,
            projection={"_id": 1, "title": 1, "created_at": 1}
        ).sort("created_at", DESCENDING).limit(limit)

        project_dicts = await cursor.to_list(length=limit)

        return [
            ProjectBaseResponse.from_mongo(project_dict)
            for project_dict in project_dicts
        ]
    except Exception as e:
        logger.error("Failed to fetch projects: %s", str(e), exc_info=True)
        raise e


async def find_project_by_id(project_id: str, username: str) -> ProjectResponse:
    """
    Find a single project by its ID'.
    Raises:
        HTTPException 400: Invalid project ID format.
        HTTPException 404: Project not found.
    """
    try:
        object_id = parse_object_id(project_id)
        project_filter = {"_id": object_id, "created_by": username}
        project_dict = await get_db().projects.find_one(project_filter)
        if not project_dict:
            raise HTTPException(status_code=404, detail=f"Project with id {project_id} not found")
        return ProjectResponse.from_mongo(project_dict)
    except Exception as e:
        logger.error("Failed to find project (project_id=%s) for user %s: %s", project_id, username, str(e),
                     exc_info=True)
        raise e


async def get_project_instruction(project_id: str, username: str) -> str | None:
    """
    Get project instruction field for use in LLM requests.
    
    Args:
        project_id: The project ID to lookup
        username: Username for ownership verification
        
    Returns:
        Project instruction string if found, None otherwise
        
    Raises:
        HTTPException 400: Invalid project ID format
        HTTPException 404: Project not found or not owned by user
    """
    try:
        object_id = parse_object_id(project_id)
        project_filter = {"_id": object_id, "created_by": username}
        
        # Only fetch the instruction field to minimize data transfer
        project_dict = await get_db().projects.find_one(
            project_filter, 
            projection={"instruction": 1}
        )
        
        if not project_dict:
            raise HTTPException(
                status_code=404, 
                detail=f"Project with id {project_id} not found or not owned by user"
            )
        
        instruction = project_dict.get("instruction")
        
        # Return instruction if it exists and is not empty
        if instruction and instruction.strip():
            logger.info(f"Retrieved project instruction for project_id={project_id}, length={len(instruction)} chars")
            return instruction.strip()
        else:
            logger.debug(f"No instruction found for project_id={project_id}")
            return None
            
    except HTTPException:
        # Re-raise HTTPExceptions as-is
        raise
    except Exception as e:
        logger.error("Failed to get project instruction (project_id=%s) for user %s: %s", 
                     project_id, username, str(e), exc_info=True)
        # Return None instead of raising to prevent breaking chat completions
        return None


async def create_project(project: ProjectModel, username: str) -> ProjectResponse:
    try:
        if not project.title or str(project.title).strip() == "":
            raise HTTPException(status_code=400, detail="Project title is required")

        project_doc = {
            **project.model_dump(),
            "created_at": datetime.now(timezone.utc),
            "created_by": username
        }
        result = await get_db().projects.insert_one(project_doc)
        logger.info("Create new project: %s by user=%s", str(result.inserted_id), username)
        project_doc["_id"] = result.inserted_id
        return ProjectResponse.from_mongo(project_doc)
    except PyMongoError as e:
        logger.exception("MongoDB error while creating project: %s", e)
        raise HTTPException(status_code=500, detail="Database error")
    except Exception as e:
        logger.exception("Unexpected error while creating project: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


async def update_project(project_id: str, project: ProjectModel, username: str) -> ProjectResponse:
    try:
        if not project.title or str(project.title).strip() == "":
            raise HTTPException(status_code=400, detail="Project title is required")

        object_id = parse_object_id(project_id)
        project_filter = {"_id": object_id, "created_by": username}

        update_fields = {
            "title": project.title,
            "description": project.description,
            "instruction": project.instruction,
            "updated_at": datetime.now(timezone.utc),
            "updated_by": username
        }

        updated_doc = await get_db().projects.find_one_and_update(
            project_filter,
            {"$set": update_fields},
            return_document=ReturnDocument.AFTER
        )
        if not updated_doc:
            raise HTTPException(status_code=404, detail="Project not found or not owned by user")
        return ProjectResponse.from_mongo(updated_doc)
    except Exception as e:
        logger.exception("Failed to update project %s: %s", project_id, e)
        raise HTTPException(status_code=400, detail=str(e))


async def patch_project(project_id: str, project: ProjectModel, username: str) -> ProjectResponse:
    try:
        object_id = parse_object_id(project_id)
        project_filter = {"_id": object_id, "created_by": username}

        update_fields = project.model_dump(exclude_unset=True)
        if not update_fields:
            raise HTTPException(status_code=400, detail="No fields provided for update")
        update_fields.update({
            "updated_at": datetime.now(timezone.utc),
            "updated_by": username
        })

        updated_doc = await get_db().projects.find_one_and_update(
            project_filter,
            {"$set": update_fields},
            return_document=ReturnDocument.AFTER
        )
        if not updated_doc:
            raise HTTPException(status_code=404, detail="Project not found or not owned by user")
        return ProjectResponse.from_mongo(updated_doc)
    except Exception as e:
        logger.exception("Failed to update project %s: %s", project_id, e)
        raise HTTPException(status_code=400, detail=str(e))


async def delete_project(project_id: str, username: str) -> None:
    db = get_db()
    try:
        object_id = parse_object_id(project_id)
        project_filter = {"_id": object_id, "created_by": username}
        project = await db.projects.find_one(project_filter)
        if not project:
            logger.exception("Project %s not found.", project_id)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail="Project not found or not owned by the user")

        chats_query = {"project_id": str(object_id)}
        attachment_filter = {"project_id": str(object_id), "meta.type": "document"}

        async def _do_delete(session=None):
            chat_ids = []
            async for doc in db.chats.find(chats_query, {"_id": 1}, session=session):
                chat_ids.append(doc["_id"])

            if chat_ids:
                chat_id_strs = [str(cid) for cid in chat_ids]
                if chat_id_strs:
                    messages_query = {"chat_id": {"$in": chat_id_strs}}
                    cursor = db.messages.find(messages_query, session=session)
                    message_dict = await cursor.to_list(length=None)
                    if message_dict:
                        await db.messages_deleted.insert_many(message_dict, session=session)
                    res_msg = await db.messages.delete_many(messages_query, session=session)
                    logger.info(
                        "Soft deleted %d messages for project %s",
                        getattr(res_msg, "deleted_count", 0),
                        project_id,
                    )

            cursor = db.chats.find(chats_query, session=session)
            chats_dict = await cursor.to_list(length=None)
            if chats_dict:
                await db.chats_deleted.insert_many(chats_dict, session=session)
            res_chats = await db.chats.delete_many(chats_query, session=session)
            logger.info(
                "Soft deleted %d chats for project %s",
                getattr(res_chats, "deleted_count", 0),
                project_id,
            )

            cursor = db.attachments.find(attachment_filter, session=session)
            attachments_dict = await cursor.to_list(length=None)
            if attachments_dict:
                await db.attachments_deleted.insert_many(attachments_dict, session=session)
            res_attachments = await db.attachments.delete_many(attachment_filter, session=session)
            logger.info(
                "Soft deleted %d attachments for project %s",
                getattr(res_attachments, "deleted_count", 0),
                project_id,
            )

            await db.projects_deleted.insert_one(project, session=session)
            res_proj = await db.projects.delete_one(project_filter, session=session)
            if getattr(res_proj, "deleted_count", 0) == 0:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found during delete")

        try:
            async with await db.client.start_session() as session:
                async with session.start_transaction():
                    await _do_delete(session=session)
        except ConfigurationError:
            logger.warning("MongoDB transactions are not supported; proceeding without a transaction.")
            await _do_delete(session=None)
        pass
    except PyMongoError as e:
        logger.exception("Database error while deleting project %s: %s", project_id, e)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database error")
    except Exception as e:
        logger.exception("Unexpected error while deleting project %s: %s", project_id, e)
        raise
