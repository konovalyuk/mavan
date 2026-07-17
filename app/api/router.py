from fastapi import APIRouter
from app.api import (auth_routes, chat_routes, domain_routes, file_routes, intelligence_routes, message_routes,
                     project_routes, prompt_routes)

api_router = APIRouter()

api_router.include_router(chat_routes.router)
api_router.include_router(message_routes.router)
api_router.include_router(prompt_routes.router)
api_router.include_router(project_routes.router)
api_router.include_router(file_routes.router)
api_router.include_router(auth_routes.router)
api_router.include_router(intelligence_routes.router)
api_router.include_router(domain_routes.router)
