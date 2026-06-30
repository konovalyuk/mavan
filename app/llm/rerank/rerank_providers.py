from dataclasses import dataclass

from config import llm_settings
from app.llm.rerank.providers.cohere import CohereRerankAdapter
from app.llm.rerank.providers.mock import MockRerankAdapter
from app.llm.rerank.types import RerankProvider

RERANK_PROVIDERS = frozenset({"mock", "cohere"})


@dataclass(frozen=True)
class RerankConfig:
    provider: str
    model: str


def rerank_provider_name(provider: str | None = None) -> str:
    name = (provider or llm_settings.RERANK_PROVIDER).lower()
    if name not in RERANK_PROVIDERS:
        raise ValueError(
            f"Provider {name!r} does not support rerank. "
            f"Supported: {', '.join(sorted(RERANK_PROVIDERS))}"
        )
    return name


def resolve_rerank_model(*, provider: str | None = None) -> str:
    if llm_settings.RERANK_MODEL:
        return llm_settings.RERANK_MODEL

    name = rerank_provider_name(provider)
    if name == "mock":
        return "mock-rerank"
    if name == "cohere":
        return llm_settings.COHERE_RERANK_MODEL
    raise ValueError(f"No default rerank model for provider: {name!r}")


def resolve_rerank_config(*, provider: str | None = None) -> RerankConfig:
    name = rerank_provider_name(provider)
    return RerankConfig(provider=name, model=resolve_rerank_model(provider=name))


def get_rerank_provider(provider: str | None = None) -> RerankProvider:
    name = rerank_provider_name(provider)

    if name == "mock":
        return MockRerankAdapter()

    if name == "cohere":
        if not llm_settings.COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY missing for rerank provider: cohere")
        return CohereRerankAdapter(api_key=llm_settings.COHERE_API_KEY)

    raise ValueError(f"No rerank adapter wired for provider: {name!r}")
