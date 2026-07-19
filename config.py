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
    default_chat_model: str
    default_embed_model: str | None = None

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
    PROVIDER = os.getenv("PROVIDER", "mock")

    CHAT_PROVIDER = os.getenv("CHAT_PROVIDER") or PROVIDER
    EMBED_PROVIDER = os.getenv("EMBED_PROVIDER") or PROVIDER
    RERANK_PROVIDER = os.getenv("RERANK_PROVIDER") or "mock"
    OCR_PROVIDER = os.getenv("OCR_PROVIDER") or PROVIDER
    SPEECH_PROVIDER = os.getenv("SPEECH_PROVIDER") or PROVIDER

    CHAT_MODEL = os.getenv("LLM_CHAT_MODEL", "")
    EMBED_MODEL = os.getenv("LLM_EMBED_MODEL", "")
    RERANK_MODEL = os.getenv("LLM_RERANK_MODEL", "")
    SPEECH_MODEL = os.getenv("LLM_SPEECH_MODEL", "")

    DEFAULT_MAX_TOKENS = int(os.getenv("LLM_DEFAULT_MAX_TOKENS", "1024"))
    DEFAULT_TEMPERATURE = float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.7"))

    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "") or None
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

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
    GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

    #Cohere
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
    COHERE_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0")

    # Mistral
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
    MISTRAL_BASE_URL = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
    MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    MISTRAL_EMBED_MODEL = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")

    # DeepSeek
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    DEEPSEEK_EMBED_MODEL = os.getenv("DEEPSEEK_EMBED_MODEL", "deepseek-embed")

    # Hugging Face
    HF_TOKEN = os.getenv("HF_TOKEN", "") or os.getenv("HUGGINGFACE_API_KEY", "")
    HF_BASE_URL = os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1")
    HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
    HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

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
    OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

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
            "openai": LlmProviderPreset(cls.OPENAI_API_KEY, cls.OPENAI_BASE_URL, default_chat_model = cls.OPENAI_MODEL, default_embed_model = cls.OPENAI_EMBED_MODEL),
            "gemini": LlmProviderPreset(cls.GOOGLE_API_KEY, cls.GOOGLE_BASE_URL, default_chat_model = cls.GEMINI_MODEL, default_embed_model = cls.GEMINI_EMBED_MODEL),
            "mistral": LlmProviderPreset(cls.MISTRAL_API_KEY, cls.MISTRAL_BASE_URL, default_chat_model = cls.MISTRAL_MODEL, default_embed_model = cls.MISTRAL_EMBED_MODEL),
            "deepseek": LlmProviderPreset(cls.DEEPSEEK_API_KEY, cls.DEEPSEEK_BASE_URL, default_chat_model = cls.DEEPSEEK_MODEL, default_embed_model = cls.DEEPSEEK_EMBED_MODEL),
            "huggingface": LlmProviderPreset(cls.HF_TOKEN, cls.HF_BASE_URL, default_chat_model = cls.HF_MODEL, default_embed_model = cls.HF_EMBED_MODEL),
            "github_models": LlmProviderPreset(
                cls.GITHUB_TOKEN, cls.GITHUB_MODELS_BASE_URL, default_chat_model = cls.GITHUB_MODELS_MODEL
            ),
            "ollama": LlmProviderPreset(cls.OLLAMA_API_KEY, cls.OLLAMA_BASE_URL, default_chat_model = cls.OLLAMA_MODEL, default_embed_model = cls.OLLAMA_EMBED_MODEL),
            "anthropic": LlmProviderPreset(cls.ANTHROPIC_API_KEY, None, default_chat_model = cls.ANTHROPIC_MODEL),
            "llama_local": LlmProviderPreset("", None, default_chat_model = cls.LLAMA_MODEL),  # api_key не нужен
        }


class RagSettings:
    RETRIEVER = os.getenv("RAG_RETRIEVER", "hybrid")  # vector | file | hybrid
    VECTOR_INDEX_PATH = os.getenv("RAG_VECTOR_INDEX_PATH", "./.rag/vector_index")
    EVAL_DIR = os.getenv("RAG_EVAL_DIR", "./data/eval")
    CONTEXT_MAX_CHARS = int(os.getenv("RAG_CONTEXT_MAX_CHARS", "8000"))
    MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0"))
    MIN_RELATIVE_SCORE = float(os.getenv("RAG_MIN_RELATIVE_SCORE", "0"))
    RERANK_ENABLED = env_bool("RAG_RERANK_ENABLED", False)
    FETCH_K = int(os.getenv("RAG_FETCH_K", "20"))
    RRF_K = int(os.getenv("RAG_RRF_K", "60"))

    @property
    def chunks_path(self) -> Path:
        return Path(self.VECTOR_INDEX_PATH) / "chunks.json"

    @property
    def ground_truth_path(self) -> str:
        return os.path.join(self.EVAL_DIR, "ground_truth.jsonl")


class DomainSettings:
    MODEL_DIR = os.getenv("DOMAIN_MODEL_DIR", str(BASE_DIR / "data" / "domain_models"))
    INGEST_CACHE_TTL_HOURS = int(os.getenv("INGEST_CACHE_TTL_HOURS", "24"))
    QUALITY_PROVIDERS = os.getenv("QUALITY_PROVIDERS", "gemini,openai,mistral")
    FORECAST_PROVIDERS = os.getenv("FORECAST_PROVIDERS") or QUALITY_PROVIDERS
    QUALITY_TIMEOUT_SEC = int(os.getenv("QUALITY_TIMEOUT_SEC", "60"))
    FORECAST_TIMEOUT_SEC = int(os.getenv("FORECAST_TIMEOUT_SEC", "60"))
    QUALITY_CYCLE_MAX = int(os.getenv("QUALITY_CYCLE_MAX", "5"))
    # Per-tick LLM tool-loop safety (not conveyor lifetime)
    SPECIALIST_AGENT_MAX_ROUNDS = int(os.getenv("SPECIALIST_AGENT_MAX_ROUNDS", "6"))
    DECISION_AGENT_MAX_ROUNDS = int(os.getenv("DECISION_AGENT_MAX_ROUNDS", "8"))
    PACKET_REFINE_MAX = int(os.getenv("PACKET_REFINE_MAX", "8"))
    COORDINATOR_PROVIDER = os.getenv("COORDINATOR_PROVIDER") or os.getenv("CHAT_PROVIDER", "gemini")
    RESEARCH_PROVIDER = os.getenv("RESEARCH_PROVIDER") or COORDINATOR_PROVIDER
    EXTRACT_PROVIDER = os.getenv("EXTRACT_PROVIDER") or COORDINATOR_PROVIDER
    TRAIN_AGENT_PROVIDER = os.getenv("TRAIN_AGENT_PROVIDER") or COORDINATOR_PROVIDER
    TRAIN_STEPS_PER_MICRO = int(os.getenv("TRAIN_STEPS_PER_MICRO", "1"))
    CORE_MEMORY_MAX_SAMPLES = int(os.getenv("CORE_MEMORY_MAX_SAMPLES", "2000"))
    QUALITY_CREDIBILITY_MIN = float(os.getenv("QUALITY_CREDIBILITY_MIN", "70"))
    QUALITY_COMPLETENESS_MIN = float(os.getenv("QUALITY_COMPLETENESS_MIN", "60"))
    QUALITY_DEPTH_MIN = float(os.getenv("QUALITY_DEPTH_MIN", "55"))
    QUALITY_TERMINOLOGY_MIN = float(os.getenv("QUALITY_TERMINOLOGY_MIN", "60"))
    QUALITY_COHERENCE_MIN = float(os.getenv("QUALITY_COHERENCE_MIN", "65"))
    QUALITY_MISUNDERSTANDING_MAX = float(os.getenv("QUALITY_MISUNDERSTANDING_MAX", "40"))
    TRAIN_MIN_SAMPLES = int(os.getenv("TRAIN_MIN_SAMPLES", "50"))
    TRAIN_MIN_UNIQUE_STATES = int(os.getenv("TRAIN_MIN_UNIQUE_STATES", "15"))
    TRAIN_MIN_UNIQUE_ACTIONS = int(os.getenv("TRAIN_MIN_UNIQUE_ACTIONS", "5"))
    TRAIN_MIN_AVG_QUALITY = float(os.getenv("TRAIN_MIN_AVG_QUALITY", "70"))
    TRAIN_REPLAY_RATIO = float(os.getenv("TRAIN_REPLAY_RATIO", "0.3"))
    GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY", "")
    GOOGLE_SEARCH_URL = os.getenv("GOOGLE_SEARCH_URL", "https://google.serper.dev/search")
    MODEL_D_MODEL = int(os.getenv("DOMAIN_MODEL_D_MODEL", "256"))
    MODEL_N_LAYERS = int(os.getenv("DOMAIN_MODEL_N_LAYERS", "2"))
    MODEL_N_HEADS = int(os.getenv("DOMAIN_MODEL_N_HEADS", "4"))
    MODEL_FFN = int(os.getenv("DOMAIN_MODEL_FFN", "512"))
    MODEL_VOCAB = int(os.getenv("DOMAIN_MODEL_VOCAB", "8000"))
    MODEL_EPOCHS = int(os.getenv("DOMAIN_MODEL_EPOCHS", "3"))
    MODEL_BATCH = int(os.getenv("DOMAIN_MODEL_BATCH", "8"))

    @classmethod
    def model_path(cls, domain_id: str) -> Path:
        return Path(cls.MODEL_DIR) / domain_id


class AgentSettings:
    AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.2"))
    AGENT_TOOL_CHOICE = os.getenv("AGENT_TOOL_CHOICE", "auto")


flask_settings = FlaskSettings()
api_settings = ApiSettings()
mongo_settings = MongoSettings()
auth_settings = AuthSettings()
file_settings = FileSettings()
llm_settings = LlmSettings()
rag_settings = RagSettings()
domain_settings = DomainSettings()
agent_settings = AgentSettings()
