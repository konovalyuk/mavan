# stdlib
import mimetypes
from typing import List, Union

# third-party
from fastapi import APIRouter, Depends, Path, Body, Query, status, UploadFile, File
from fastapi.responses import FileResponse

from app.models.attachment_model import FileAttachmentResponse, DocumentAttachmentResponse
# internal/local
from app.services.auth_service import get_user_from_token
from app.models.auth_model import MavanUser
from app.services.file_service import upload_files, download_file, delete_file, get_attachments_metadata

router = APIRouter()


@router.post("/api/v1/files/upload",
             summary="Upload a file",
             response_model=List[FileAttachmentResponse],
             status_code=status.HTTP_201_CREATED,
             response_model_exclude_none=True,
             responses={400: {"description": "Bad request"},
                        401: {"description": "Unauthorized"},
                        413: {"description": "Payload too large"},
                        500: {"description": "Internal Server Error"}
                        })
async def api_upload_files(
        files: list[UploadFile] = File(..., description="Files to upload"),
        current_user: MavanUser = Depends(get_user_from_token)
) -> list[FileAttachmentResponse]:
    """Endpoint to upload a file."""
    return await upload_files(files, current_user.username)


@router.get("/api/v1/files/{attachment_id}/download",
            summary="Download a file",
            status_code=status.HTTP_200_OK,
            response_model_exclude_none=True,
            responses={400: {"description": "Bad request"},
                       401: {"description": "Unauthorized"},
                       404: {"description": "File not found"},
                       500: {"description": "Internal Server Error"}
                       })
async def api_download_file(
        attachment_id: str = Path(..., description="Path of the file to download"),
        current_user: MavanUser = Depends(get_user_from_token)
):
    """Endpoint to download a file."""
    file: Path = await download_file(attachment_id, current_user.username)
    mime_type, _ = mimetypes.guess_type(str(file))
    media_type = mime_type or "application/octet-stream"
    return FileResponse(
        path=str(file),
        filename=file.name,
        media_type=media_type
    )


@router.get("/api/v1/attachments/metadata",
            summary="Get attachment metadata",
            response_model=list[Union[FileAttachmentResponse, DocumentAttachmentResponse]],
            status_code=status.HTTP_200_OK,
            response_model_exclude_none=True,
            responses={400: {"description": "Bad request"},
                       401: {"description": "Unauthorized"},
                       404: {"description": "File not found"},
                       500: {"description": "Internal Server Error"}
                       })
async def api_get_attachments_metadata(
        id: list[str] = Query(..., description="List of attachment IDs to get metadata for"),
        current_user: MavanUser = Depends(get_user_from_token)
) -> list[Union[FileAttachmentResponse, DocumentAttachmentResponse]]:
    """Endpoint to get file metadata."""
    return await get_attachments_metadata(id, current_user.username)


@router.delete("/api/v1/files/{attachment_id}/delete",
               summary="Delete a file",
               status_code=status.HTTP_204_NO_CONTENT,
               response_model_exclude_none=True,
               responses={400: {"description": "Bad request"},
                          401: {"description": "Unauthorized"},
                          404: {"description": "File not found"},
                          500: {"description": "Internal Server Error"}
                          })
async def api_delete_file(
        attachment_id: str = Path(..., description="Path of the file to delete"),
        current_user: MavanUser = Depends(get_user_from_token)
):
    """Endpoint to delete a file."""
    await delete_file(attachment_id, current_user.username)
