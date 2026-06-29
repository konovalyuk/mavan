import os
from pathlib import Path

from dotenv import load_dotenv
from dataclasses import dataclass

# Load .env from project root
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

@dataclass(frozen=True)
class LlmProviderPreset:
    api_key: str
    base_url: str | None
    default_model: str

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
    PROVIDER = os.getenv("LLM_PROVIDER", "mock")
    CHAT_PROVIDER = os.getenv("LLM_CHAT_PROVIDER") or PROVIDER
    EMBED_PROVIDER = os.getenv("LLM_EMBED_PROVIDER") or PROVIDER
    OCR_PROVIDER = os.getenv("LLM_OCR_PROVIDER") or PROVIDER
    SPEECH_PROVIDER = os.getenv("LLM_SPEECH_PROVIDER") or PROVIDER
    CHAT_MODEL = os.getenv("LLM_CHAT_MODEL", "")
    EMBED_MODEL = os.getenv("LLM_EMBED_MODEL", "")
    DEFAULT_MAX_TOKENS = int(os.getenv("LLM_DEFAULT_MAX_TOKENS", "1024"))
    DEFAULT_TEMPERATURE = float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.7"))

    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "") or None
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Anthropic
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

    # Google / Gemini
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
    GOOGLE_BASE_URL = os.getenv(
        "GOOGLE_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

    # Mistral
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
    MISTRAL_BASE_URL = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
    MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

    # DeepSeek
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # Hugging Face
    HF_TOKEN = os.getenv("HF_TOKEN", "") or os.getenv("HUGGINGFACE_API_KEY", "")
    HF_BASE_URL = os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1")
    HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

    # GitHub Models
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "") or os.getenv("GITHUB_MODELS_TOKEN", "")
    GITHUB_MODELS_BASE_URL = os.getenv(
        "GITHUB_MODELS_BASE_URL",
        "https://models.github.ai/inference",
    )
    GITHUB_MODELS_MODEL = os.getenv("GITHUB_MODELS_MODEL", "openai/gpt-4o")

    # Ollama
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

    # Local Llama (aic-style)
    LLAMA_CKPT_DIR = os.getenv(
        "LLAMA_CKPT_DIR",
        "/home/maksym/.llama/checkpoints/Llama3.2-1B-Instruct",
    )
    LLAMA_MAX_SEQ_LEN = int(os.getenv("LLAMA_MAX_SEQ_LEN", "1024"))
    LLAMA_MAX_BATCH_SIZE = int(os.getenv("LLAMA_MAX_BATCH_SIZE", "1"))
    LLAMA_MODEL_PARALLEL_SIZE = int(os.getenv("LLAMA_MODEL_PARALLEL_SIZE", "1"))
    LLAMA_MAX_GEN_LEN = int(os.getenv("LLAMA_MAX_GEN_LEN", "256"))
    LLAMA_MAX_DIALOG_LENGTH = int(os.getenv("LLAMA_MAX_DIALOG_LENGTH", "1024"))
    LLAMA_MODEL = os.getenv("LLAMA_MODEL", "Llama3.2-1B-Instruct")

    @classmethod
    def provider_presets(cls) -> dict[str, LlmProviderPreset]:
        return {
            "openai": LlmProviderPreset(cls.OPENAI_API_KEY, cls.OPENAI_BASE_URL, cls.OPENAI_MODEL),
            "gemini": LlmProviderPreset(cls.GOOGLE_API_KEY, cls.GOOGLE_BASE_URL, cls.GEMINI_MODEL),
            "mistral": LlmProviderPreset(cls.MISTRAL_API_KEY, cls.MISTRAL_BASE_URL, cls.MISTRAL_MODEL),
            "deepseek": LlmProviderPreset(cls.DEEPSEEK_API_KEY, cls.DEEPSEEK_BASE_URL, cls.DEEPSEEK_MODEL),
            "huggingface": LlmProviderPreset(cls.HF_TOKEN, cls.HF_BASE_URL, cls.HF_MODEL),
            "github_models": LlmProviderPreset(
                cls.GITHUB_TOKEN, cls.GITHUB_MODELS_BASE_URL, cls.GITHUB_MODELS_MODEL
            ),
            "ollama": LlmProviderPreset(cls.OLLAMA_API_KEY, cls.OLLAMA_BASE_URL, cls.OLLAMA_MODEL),
            "anthropic": LlmProviderPreset(cls.ANTHROPIC_API_KEY, None, cls.ANTHROPIC_MODEL),
            "llama_local": LlmProviderPreset("", None, cls.LLAMA_MODEL),  # api_key не нужен
        }


flask_settings = FlaskSettings()
api_settings = ApiSettings()
mongo_settings = MongoSettings()
auth_settings = AuthSettings()
file_settings = FileSettings()
llm_settings = LlmSettings()
