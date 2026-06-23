# stdlib
from typing import List

# third-party
from fastapi import APIRouter, Depends, Path, Body, Query, status
from fastapi.responses import JSONResponse

# internal/local
from app.services.auth_service import get_user_from_token
from app.models.auth_model import MavanUser
from app.models.project_model import ProjectModel, ProjectResponse, ProjectBaseResponse
from app.services.project_service import find_all_projects, create_project, update_project, delete_project, \
    find_project_by_id, patch_project

router = APIRouter()


@router.get("/api/v1/projects", summary="Get paginated list of user projects sorted by created_at DESC",
            response_model=List[ProjectBaseResponse],
            status_code=status.HTTP_200_OK,
            response_model_exclude_none=True,
            responses={400: {"description": "Invalid data"}, 500: {"description": "Internal Server Error"}})
async def api_find_all_projects(
        limit: int = Query(20, ge=1, le=100),
        before: str | None = Query(None,
                                   description="RFC3339 timestamp (ISO 8601) to paginate before, e.g. 2025-07-02T14:59:49.508Z"),
        current_user: MavanUser = Depends(get_user_from_token)) -> List:
    return await find_all_projects(current_user.username, limit, before)


@router.get("/api/v1/projects/{project_id}", summary="Get project by id", response_model=ProjectResponse,
            status_code=status.HTTP_200_OK,
            response_model_exclude_none=True,
            responses={400: {"description": "Invalid data"},
                       404: {"description": "Project not found"},
                       500: {"description": "Internal Server Error"}})
async def api_find_project_by_id(
        project_id: str = Path(..., description="ID of the project"),
        current_user: MavanUser = Depends(get_user_from_token)):
    return await find_project_by_id(project_id, current_user.username)


@router.post("/api/v1/projects", summary="Create project", response_model=ProjectResponse,
             status_code=status.HTTP_201_CREATED,
             response_model_exclude_none=True,
             responses={400: {"description": "Invalid data"},
                        500: {"description": "Internal Server Error"}})
async def api_create_project(
        project: ProjectModel = Body(..., description="New project"),
        current_user: MavanUser = Depends(get_user_from_token)
):
    created_project = await create_project(project, current_user.username)
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content=created_project.model_dump(),
        headers={"Location": f"/api/v1/projects/{created_project.id}"}
    )


@router.put("/api/v1/projects/{project_id}", summary="Update project by id", response_model=ProjectResponse,
            status_code=status.HTTP_200_OK,
            responses={400: {"description": "Invalid data"},
                       404: {"description": "Project not found"},
                       500: {"description": "Internal Server Error"}})
async def api_update_project(
        project_id: str = Path(..., description="ID of the project"),
        project: ProjectModel = Body(..., description="Updated project"),
        current_user: MavanUser = Depends(get_user_from_token)
):
    return await update_project(project_id, project, current_user.username)


@router.patch("/api/v1/projects/{project_id}", summary="Patch project by id", response_model=ProjectResponse,
              status_code=status.HTTP_200_OK,
              response_model_exclude_none=True,
              responses={400: {"description": "Invalid data"},
                         404: {"description": "Project not found"},
                         500: {"description": "Internal Server Error"}})
async def api_patch_project(
        project_id: str = Path(..., description="ID of the project"),
        project: ProjectModel = Body(..., description="Fields to update"),
        current_user: MavanUser = Depends(get_user_from_token)
):
    return await patch_project(project_id, project, current_user.username)


@router.delete("/api/v1/projects/{project_id}", summary="Delete project",
               status_code=status.HTTP_204_NO_CONTENT,
               responses={400: {"description": "Invalid data"},
                          404: {"description": "Project not found"},
                          500: {"description": "Internal Server Error"}})
async def api_delete_project(
        project_id: str = Path(..., description="ID of the project"),
        current_user: MavanUser = Depends(get_user_from_token)
):
    await delete_project(project_id, current_user.username)
