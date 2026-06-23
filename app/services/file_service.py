import logging
import os
from typing import Union
import aiofiles
import time
from datetime import datetime, timezone

from fastapi import UploadFile, HTTPException, status
from pathlib import Path

from pymongo import ReturnDocument
from pymongo.errors import PyMongoError

from config import file_settings
from app.database import get_db
from app.models.attachment_model import AttachmentBaseModel, FileModel, FileAttachmentResponse, \
    DocumentAttachmentResponse
from app.services.util_service import parse_object_id

logger = logging.getLogger(__name__)


async def upload_files(files: list[UploadFile], username: str) -> list[FileAttachmentResponse]:
    """
    Save uploaded files to filesystem under {filesystem_path}/{username}/
    Streaming write with per-file size enforcement.
    """
    try:
        MAX_FILES_PER_REQUEST = file_settings.MAX_FILES_PER_REQUEST
        MAX_FILE_SIZE_BYTES = file_settings.MAX_FILE_SIZE_BYTES
        ALLOWED_FILE_TYPES = [ft.strip() for ft in file_settings.ALLOWED_FILE_TYPES.split(",")]

        if len(files) > MAX_FILES_PER_REQUEST:
            logger.error("Too many files in request: %d", len(files))
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"Too many files. Maximum allowed is {MAX_FILES_PER_REQUEST}.")

        filesystem_path = Path(file_settings.FILESYSTEM_PATH).resolve()
        if not filesystem_path.exists() or not filesystem_path.is_dir():
            logger.error("Filesystem path is not configured correctly: %s", str(filesystem_path))
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Filesystem path is not configured correctly on the server.")
        upload_dir = (filesystem_path / username).resolve()
        try:
            upload_dir.relative_to(filesystem_path)
        except Exception:
            logger.error("Upload directory is outside of filesystem path: %s", str(upload_dir))
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Server configuration error (filesystem path).")
        upload_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []
        db = get_db()
        for file in files:
            if file.content_type not in ALLOWED_FILE_TYPES:
                logger.error("File type not allowed: %s", file.content_type)
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                    detail=f"File type '{file.content_type}' of file '{file.filename}' is not allowed.")

            filename = Path(file.filename).name  # sanitize filename
            if not filename:
                logger.error("Invalid file name: %s", file.filename)
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                    detail="Invalid file name.")

            size_known = None
            try:
                file.file.seek(0, os.SEEK_END)
                size_known = file.file.tell()
                file.file.seek(0)
            except Exception:
                size_known = None

            if size_known is not None and size_known > MAX_FILE_SIZE_BYTES:
                logger.error("File size exceeds limit: %s (%d bytes)", filename, size_known)
                raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                    detail=f"File '{filename}' exceeds the maximum size of {MAX_FILE_SIZE_BYTES} bytes.")

            file_model = FileModel(
                type="file",
                file_name=filename,
                file_type=file.content_type
            )
            attachment_model = AttachmentBaseModel(
                state="uploading",
                meta=file_model,
                created_at=datetime.now(timezone.utc),
                created_by=username
            )
            attachment_doc = attachment_model.model_dump(exclude_none=True)
            try:
                result = await db.attachments.insert_one(attachment_doc)
                logger.info("Start uploading new file: %s by user=%s", str(result.inserted_id), username)
                attachment_doc["_id"] = result.inserted_id
            except PyMongoError as e:
                logger.error(f"MongoDB error while saving file metadata: {e}")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                    detail="Database error while saving file metadata.")

            file_path = get_unique_file_path(upload_dir, filename, str(attachment_doc["_id"]))
            try:
                file_path.relative_to(filesystem_path)
            except Exception:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                    detail="Server configuration error (filesystem path).")

            await save_upload_file_streaming(file, file_path, MAX_FILE_SIZE_BYTES)
            file_filter = {"_id": attachment_doc["_id"]}

            file_model.file_path = str(file_path.relative_to(filesystem_path))
            file_model.file_size = os.path.getsize(file_path)
            attachment_model.meta = file_model
            attachment_model.state = "uploaded"
            attachment_model.updated_at = datetime.now(timezone.utc)
            attachment_model.updated_by = username

            try:
                result = await db.attachments.find_one_and_update(
                    file_filter,
                    {"$set": attachment_model.model_dump(exclude_none=True)},
                    return_document=ReturnDocument.AFTER
                )
                if not result:
                    logger.error("Failed to update file metadata after upload: %s", str(attachment_doc["_id"]))
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                        detail="Database error while updating file metadata.")
                logger.info("Upload new file: %s by user=%s", str(attachment_doc["_id"]), username)
                flat = flatten_attachment_doc(result)
                saved_files.append(FileAttachmentResponse.model_validate(flat))
            except PyMongoError as e:
                logger.error(f"MongoDB error while saving file metadata: {e}")
                try:
                    file_path.unlink(missing_ok=True)
                except Exception:
                    logger.warning("Failed to remove file after DB error: %s", file_path)
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                    detail="Database error while saving file metadata.")
        return saved_files
    except Exception as e:
        logger.error(f"Error uploading files: {e}")
        raise e


async def save_upload_file_streaming(file: UploadFile, file_path: Path, max_file_size: int):
    """
    Read upload in chunks (async) and write to disk. If the accumulated size exceeds max_file_size,
    remove partial file and raise HTTPException(413).
    """
    CHUNK_SIZE = 1024 * 1024  # 1 MB
    total_bytes = 0
    tmp_path = file_path.with_suffix(file_path.suffix + f".tmp-{os.getpid()}-{int(time.time() * 1000)}")
    try:
        async with aiofiles.open(tmp_path, "wb") as buffer:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_file_size:
                    await buffer.close()
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        logger.warning("Failed to remove partial file %s", file_path)
                    raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                        detail=f"File '{file.filename}' exceeds the maximum size of {max_file_size} bytes.")
                await buffer.write(chunk)
        os.replace(tmp_path, file_path)
    finally:
        try:
            await file.close()
        except Exception:
            pass
        if tmp_path.exists():
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                logger.debug("Failed to cleanup tmp file %s", tmp_path)


def get_unique_file_path(upload_dir: Path, filename: str, id: str) -> Path:
    base = Path(filename).stem
    suffix = Path(filename).suffix
    unique_name = f"{base}_{id}{suffix}"
    return upload_dir / unique_name


async def download_file(attachment_id: str, username: str) -> Path:
    try:
        object_id = parse_object_id(attachment_id)
        attachment_filter = {"_id": object_id, "created_by": username, "state": "uploaded"}
        attachment_dict = await get_db().attachments.find_one(attachment_filter)
        if not attachment_dict:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Attachment not found or not owned by user.")
        attachment_type = attachment_dict.get("meta").get("type")
        if attachment_type != "file":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Attachment is not a file.")
        file_path = attachment_dict.get("meta").get("file_path")
        if not file_path:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File path not found in database.")
        return get_path_if_exists(file_path)
    except PyMongoError as e:
        logger.error(f"MongoDB error while fetching file metadata: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error while fetching file metadata.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files with this id.")


async def get_attachments_metadata(attachment_ids: list[str], username: str) -> list[Union[FileAttachmentResponse, DocumentAttachmentResponse]]:
    try:
        object_ids = [parse_object_id(aid) for aid in attachment_ids]
        attachment_filter = {
            "_id": {"$in": object_ids},
            "created_by": username,
            "state": {"$ne": "deleted"}
        }
        cursor = get_db().attachments.find(attachment_filter)
        attachments = []
        async for attachment_dict in cursor:
            flat = flatten_attachment_doc(attachment_dict)
            if flat.get("type") == "file":
                attachments.append(FileAttachmentResponse.model_validate(flat))
            elif flat.get("type") == "document":
                attachments.append(DocumentAttachmentResponse.model_validate(flat))
            else:
                attachments.append(flat)
        return attachments
    except PyMongoError as e:
        logger.error(f"MongoDB error while fetching files metadata: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Database error while fetching files metadata.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error retrieving files metadata.")


async def delete_file(attachment_id: str, username: str):
    try:
        object_id = parse_object_id(attachment_id)
        db = get_db()

        attachment_filter = {"_id": object_id, "created_by": username}
        attachment_dic = await db.attachments.find_one(attachment_filter)

        if not attachment_dic:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Attachment not found or not owned by user.")

        if attachment_dic.get("chat_id"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete a attachment that is attached to a chat.")
        if attachment_dic.get("project_id"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete a attachment that is attached to a project.")

        if attachment_dic.get("meta").get("type") == "file":
            file_path = attachment_dic.get("meta").get("file_path")
            if not file_path:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File path not found in database.")
            path_to_delete = get_path_if_exists(file_path)
            try:
                path_to_delete.unlink(missing_ok=True)
            except Exception:
                logger.error("Failed to delete file %s", file_path)
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error deleting file from filesystem.")

        await db.attachments.delete_one(attachment_filter)
        logger.info("Delete attachment %s from collection attachments", attachment_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error deleting file %s: %s", attachment_id, e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files with this id.")


async def soft_delete_attachments(db, session, chat_id: str, username: str):
    """
    Soft delete attachments for a chat by moving them to attachments_deleted collection.
    This is part of the transactional chat deletion process.
    """
    try:
        attachment_filter = {"chat_id": chat_id, "created_by": username}
        cursor = db.attachments.find(attachment_filter, session=session)
        attachments_dict = await cursor.to_list(length=None)
        if attachments_dict:
            await db.attachments_deleted.insert_many(attachments_dict, session=session)
            await db.attachments.delete_many(attachment_filter, session=session)
            logger.info("Soft deleted %d attachments for chat %s from collection attachments", len(attachments_dict), chat_id)
        else:
            logger.info("No attachments found for chat %s to delete", chat_id)
    except Exception as e:
        logger.exception("Error soft deleting attachments for chat %s: %s", chat_id, e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error deleting attachments for the chat.")


def get_path_if_exists(file_path: str) -> Path:
    filesystem_path = Path(file_settings.FILESYSTEM_PATH).resolve()
    full_path = (filesystem_path / file_path).resolve()
    try:
        full_path.relative_to(filesystem_path)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid file path.")
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")
    return full_path


def flatten_attachment_doc(attachment_doc: dict) -> dict:
    """
    Converts a Mongo attachment doc into a flat structure for the response.
    Doesn't perform additional validation—that's the job of Pydantic response models.
    """
    if attachment_doc is None:
        return {}

    meta = attachment_doc.get("meta") or {}
    if hasattr(meta, "model_dump"):
        meta = meta.model_dump(exclude_none=True)
    elif not isinstance(meta, dict):
        try:
            meta = dict(meta)
        except Exception:
            meta = {}

    base = {"id": str(attachment_doc.get("_id"))}

    type = meta.get("type")
    if type == "file":
        return {
            **base,
            "type": "file",
            "file_name": meta.get("file_name"),
            "file_type": meta.get("file_type")
        }
    elif type == "document":
        return {
            **base,
            "type": "document",
            "document_id": meta.get("document_id"),
            "subject": meta.get("subject"),
            "author": meta.get("author"),
            "document_updated_at": meta.get("document_updated_at"),
            # optionally include project_id from top-level attachment_doc
            "project_id": attachment_doc.get("project_id"),
            "status": attachment_doc.get("state"),
            "updated_at": attachment_doc.get("updated_at"),
        }
    else:
        return {
            **base,
            "meta": meta,
            "state": attachment_doc.get("state"),
        }
