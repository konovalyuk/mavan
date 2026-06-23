import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api import router
from app.database import create_indexes, get_client

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    await create_indexes()
    logger.info("MongoDB indexes created")
    yield
    # shutdown
    client = get_client()
    client.close()
    logger.info("MongoDB connection closed")


app = FastAPI(title="Mavan API", lifespan=lifespan)
app.include_router(router.api_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
