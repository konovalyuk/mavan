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


class MongoSettings:
    URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    DATABASE = os.getenv("MONGODB_DATABASE", "mavan")


class AuthSettings:
    MODE = os.getenv("AUTH_MODE", "dev")  # dev | jwt
    DEV_USERNAME = os.getenv("DEV_USERNAME", "konovalyuk")

    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", os.getenv("SECRET_KEY", "dev-secret-key"))
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))

    # Single user without Mongo — for AUTH_MODE=jwt
    LOGIN_USERNAME = os.getenv("LOGIN_USERNAME", "konovalyuk")
    LOGIN_PASSWORD = os.getenv("LOGIN_PASSWORD", "")  # only for dev/single-user


class FileSettings:
    FILESYSTEM_PATH = os.getenv("FILESYSTEM_PATH", "./data/uploads")
    MAX_FILES_PER_REQUEST = int(os.getenv("MAX_FILES_PER_REQUEST", "10"))
    MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE_BYTES", str(10 * 1024 * 1024)))
    ALLOWED_FILE_TYPES = os.getenv(
        "ALLOWED_FILE_TYPES",
        "pdf,txt,md,doc,docx,xls,xlsx,ppt,pptx,csv,json",
    )


class LlmSettings:
    PROVIDER = os.getenv("LLM_PROVIDER", "mock")  # mock | openai | ollama | gemini
    MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")  # optional
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    DEFAULT_MAX_TOKENS = int(os.getenv("LLM_DEFAULT_MAX_TOKENS", "1024"))
    DEFAULT_TEMPERATURE = float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.7"))


flask_settings = FlaskSettings()
api_settings = ApiSettings()
mongo_settings = MongoSettings()
auth_settings = AuthSettings()
file_settings = FileSettings()
llm_settings = LlmSettings()
