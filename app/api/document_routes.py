# stdlib
from typing import List

# third-party
from fastapi import APIRouter, Depends, Body, status

from app.models.attachment_model import DocumentModel, DocumentAttachmentResponse
# internal/local
from app.services.auth_service import get_user_from_token
from app.models.auth_model import MavanUser
from app.services.document_service import save_document

router = APIRouter()


@router.post("/api/v1/documents",
             summary="Save document",
             response_model=List[DocumentAttachmentResponse],
             status_code=status.HTTP_201_CREATED,
             response_model_exclude_none=True,
             responses={400: {"description": "Bad request"},
                        401: {"description": "Unauthorized"},
                        413: {"description": "Payload too large"},
                        500: {"description": "Internal Server Error"}
                        })
async def api_save_document(
        documents: list[DocumentModel] = Body(..., description="List of documents to save"),
        current_user: MavanUser = Depends(get_user_from_token)
) -> list[DocumentAttachmentResponse]:
    """Endpoint to save documents."""
    return await save_document(documents, current_user.username)
