import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class FlaskSettings:
    HOST = os.getenv("FLASK_HOST", "127.0.0.1")
    PORT = int(os.getenv("FLASK_PORT", "5000"))
    DEBUG = env_bool("FLASK_DEBUG", True)
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")


class ApiSettings:
    HOST = os.getenv("API_HOST", "127.0.0.1")
    PORT = int(os.getenv("API_PORT", "8000"))
    RELOAD = env_bool("API_RELOAD", True)


flask_settings = FlaskSettings()
api_settings = ApiSettings()