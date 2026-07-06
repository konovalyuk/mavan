import os
from pathlib import Path

from config import BASE_DIR


class DomainSettings:
    MODEL_DIR = os.getenv("DOMAIN_MODEL_DIR", str(BASE_DIR / "data" / "domain_models"))
    QUALITY_PROVIDERS = os.getenv("QUALITY_PROVIDERS", "gemini,openai,mistral")
    FORECAST_PROVIDERS = os.getenv("FORECAST_PROVIDERS") or QUALITY_PROVIDERS
    QUALITY_TIMEOUT_SEC = int(os.getenv("QUALITY_TIMEOUT_SEC", "60"))
    FORECAST_TIMEOUT_SEC = int(os.getenv("FORECAST_TIMEOUT_SEC", "60"))
    QUALITY_CYCLE_MAX = int(os.getenv("QUALITY_CYCLE_MAX", "5"))
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


domain_settings = DomainSettings()
