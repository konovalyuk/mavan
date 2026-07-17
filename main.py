"""FastAPI entry point. Keeps PyCharm conf: uvicorn main:app"""
from app.main import app

__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn
    from config import api_settings

    uvicorn.run(
        "main:app",
        host=api_settings.HOST,
        port=api_settings.PORT,
        reload=api_settings.RELOAD,
    )