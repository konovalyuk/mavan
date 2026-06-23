from fastapi import APIRouter
from app.api import chat_routes, message_routes, prompt_routes, project_routes, file_routes, document_routes, \
    auth_routes, chat_completions_routes

api_router = APIRouter()

api_router.include_router(chat_routes.router)
api_router.include_router(message_routes.router)
api_router.include_router(prompt_routes.router)
api_router.include_router(project_routes.router)
api_router.include_router(file_routes.router)
api_router.include_router(document_routes.router)
api_router.include_router(auth_routes.router)
api_router.include_router(chat_completions_routes.router)
