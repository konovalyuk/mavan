import logging
from pathlib import Path

from fastapi import HTTPException

from app.database import get_db
from app.rag.index_store import upsert_chunks
from app.rag.pipeline import reset_pipeline
from app.rag.sources import attachment_remove_prefixes, attachment_source_prefix
from app.rag.stores.file_store import TEXT_EXTENSIONS, chunk_text
from app.services.file_service import get_path_if_exists
from app.services.util_service import parse_object_id

logger = logging.getLogger(__name__)


async def _load_attachment_for_index(attachment_id: str, *, username: str) -> Path:
    doc = await get_db().attachments.find_one({"_id": parse_object_id(attachment_id), "created_by": username})
    if not doc:
        raise HTTPException(404, f"Attachment {attachment_id} not found")
    if doc.get("state") not in ("uploaded", "indexed", "index_failed"):
        raise HTTPException(400, f"Attachment {attachment_id} in state {doc.get('state')!r}")
    rel = (doc.get("meta") or {}).get("file_path")
    if not rel:
        raise HTTPException(400, "File path missing")
    path = get_path_if_exists(rel)
    if path.suffix.lower() not in TEXT_EXTENSIONS:
        raise HTTPException(400, "Only .txt and .md supported")
    return path


async def unindex_attachments(attachment_ids: list[str], *, username: str) -> dict:
    if not attachment_ids:
        raise HTTPException(400, "attachment_ids required")

    stats = await upsert_chunks([], remove_prefixes=attachment_remove_prefixes(attachment_ids))
    for aid in attachment_ids:
        await get_db().attachments.update_one(
            {"_id": parse_object_id(aid), "created_by": username},
            {"$set": {"state": "uploaded", "updated_by": username}, "$unset": {"rag_chunk_count": "", "index_error": ""}},
        )
    reset_pipeline()
    return {"state": "uploaded", **stats}


async def index_attachments(attachment_ids: list[str], *, username: str) -> dict:
    if not attachment_ids:
        raise HTTPException(400, "attachment_ids required")

    remove_prefixes = attachment_remove_prefixes(attachment_ids)
    new_chunks = []
    per_file: list[dict] = []
    loaded: list[tuple[str, Path]] = []

    for aid in attachment_ids:
        path = await _load_attachment_for_index(aid, username=username)
        await get_db().attachments.update_one(
            {"_id": parse_object_id(aid)},
            {"$set": {"state": "indexing", "updated_by": username}},
        )
        loaded.append((aid, path))

    try:
        for aid, path in loaded:
            source = f"{attachment_source_prefix(aid)}:{path.name}"
            text = path.read_text(encoding="utf-8", errors="replace")
            file_chunks = chunk_text(text, source=source)
            if not file_chunks:
                raise ValueError(f"No chunks from attachment {aid}")
            new_chunks.extend(file_chunks)
            per_file.append({"attachment_id": aid, "chunks": len(file_chunks)})

        stats = await upsert_chunks(new_chunks, remove_prefixes=remove_prefixes)

        for aid in attachment_ids:
            count = next(x["chunks"] for x in per_file if x["attachment_id"] == aid)
            await get_db().attachments.update_one(
                {"_id": parse_object_id(aid)},
                {"$set": {"state": "indexed", "rag_chunk_count": count, "updated_by": username}},
            )

        reset_pipeline()
        return {"state": "indexed", "files": per_file, **stats}

    except Exception as e:
        for aid in attachment_ids:
            await get_db().attachments.update_one(
                {"_id": parse_object_id(aid)},
                {"$set": {"state": "index_failed", "index_error": str(e), "updated_by": username}},
            )
        raise HTTPException(500, f"Indexing failed: {e}") from e
