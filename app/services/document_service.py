import logging
from datetime import datetime, timezone

from fastapi import HTTPException
from pymongo import ReturnDocument, UpdateMany
from pymongo.errors import PyMongoError, ConfigurationError

from app.database import get_db, get_client
from app.models.attachment_model import DocumentModel, DocumentAttachmentResponse
from app.services.file_service import flatten_attachment_doc

logger = logging.getLogger(__name__)


async def save_document(documents: list[DocumentModel], username: str) -> list[DocumentAttachmentResponse]:
    """
       Saves/updates the document list as attachments.meta.type=document.
        - For incoming document_ids: updates/creates records (upsert).
        - For database records whose document_id is not in the incoming list: marks state='deleted'.
        Returns a DocumentAttachmentResponse list of modified/created records (in the order of the input list).
       """
    if documents is None:
        raise HTTPException(status_code=400, detail="No documents provided")

    db = get_db()
    client = get_client()
    now = datetime.now(timezone.utc)

    incoming_by_id: dict[int, DocumentModel] = {}
    incoming_ids: list[int] = []
    for document in documents:
        if getattr(document, "type", None) != "document":
            raise HTTPException(status_code=400,
                                detail=f"Invalid document type for document_id={getattr(document, 'document_id', None)}")
        incoming_by_id[document.document_id] = document
        incoming_ids.append(document.document_id)
    logger.info("Processing %d documents for user %s: %s", len(incoming_ids), username, incoming_ids)

    if not incoming_ids:
        raise HTTPException(status_code=400, detail="No document ids provided")

    results: list[DocumentAttachmentResponse] = []

    async def _do_work(session=None):
        for doc in documents:
            meta_dict = doc.model_dump(exclude_none=True)  # contains type=document and other fields
            filter_query = {
                "created_by": username,
                "meta.type": "document",
                "meta.document_id": meta_dict["document_id"],
                "chat_id": {"$exists": False},
                "project_id": {"$exists": False}
            }

            update_doc = {
                "$set": {
                    "meta": meta_dict,
                    "state": "available",
                    "updated_at": now,
                    "updated_by": username
                },
                "$setOnInsert": {
                    "created_at": now,
                    "created_by": username
                }
            }

            updated = await db.attachments.find_one_and_update(
                filter_query,
                update_doc,
                upsert=True,
                return_document=ReturnDocument.AFTER,
                session=session
            )

            if not updated:
                logger.error("Failed upsert for document_id=%s user=%s", meta_dict["document_id"], username)
                raise HTTPException(status_code=500, detail="Database error while upserting document attachment")

            try:
                flat = flatten_attachment_doc(updated)
                results.append(DocumentAttachmentResponse.model_validate(flat))
            except Exception:
                logger.exception("Failed to convert updated attachment to response for document_id=%s",
                                 meta_dict["document_id"])
                raise HTTPException(status_code=500, detail="Server error while preparing response")

    try:
        async with await client.start_session() as session:
            try:
                async with session.start_transaction():
                    await _do_work(session=session)
            except ConfigurationError:
                logger.warning("MongoDB transactions not supported, proceeding without transaction")
                await _do_work(session=None)
    except ConfigurationError:
        logger.warning("MongoDB transactions not supported (start_session failed), proceeding without session")
        await _do_work(session=None)
    except PyMongoError as e:
        logger.exception("Database error in save_document: %s", e)
        raise HTTPException(status_code=500, detail="Database error")
    except Exception as e:
        logger.exception("Unexpected error in save_document: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

    logger.info("Successfully saved %d documents for user %s", len(results), username)
    return results

