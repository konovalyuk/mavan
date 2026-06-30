from dataclasses import dataclass

from config import llm_settings
from app.llm.embed.providers.mock import MockEmbedAdapter
from app.llm.embed.providers.openai_compatible import OpenAICompatibleEmbedAdapter
from app.llm.embed.types import EmbedProvider

EMBED_PROVIDERS = frozenset({
    "mock", "openai", "gemini", "mistral", "deepseek", "huggingface", "ollama",
})

OPENAI_COMPATIBLE_EMBED = frozenset({
    "openai", "gemini", "mistral", "deepseek", "huggingface", "ollama",
})


@dataclass(frozen=True)
class EmbedConfig:
    provider: str
    model: str


def embed_provider_name(provider: str | None = None) -> str:
    name = (provider or llm_settings.EMBED_PROVIDER).lower()
    if name not in EMBED_PROVIDERS:
        raise ValueError(f"Provider {name!r} does not support embed. Supported: {', '.join(sorted(EMBED_PROVIDERS))}")
    return name


def resolve_embed_model(*, provider: str | None = None) -> str:
    if llm_settings.EMBED_MODEL:
        return llm_settings.EMBED_MODEL

    name = embed_provider_name(provider)
    if name == "mock":
        return "mock-embed"
    preset = llm_settings.provider_presets()[name]
    if not preset.default_embed_model:
        raise ValueError(f"Provider {name!r} has no default embed model configured")
    return preset.default_embed_model


def resolve_embed_config(*, provider: str | None = None) -> EmbedConfig:
    name = embed_provider_name(provider)
    return EmbedConfig(provider=name, model=resolve_embed_model(provider=name))


def get_embed_provider(provider: str | None = None) -> EmbedProvider:
    name = embed_provider_name(provider)

    if name == "mock":
        return MockEmbedAdapter()

    preset = llm_settings.provider_presets()[name]
    if name in OPENAI_COMPATIBLE_EMBED:
        if not preset.api_key and name != "ollama":
            raise ValueError(f"API key missing for embed provider: {name}")
        return OpenAICompatibleEmbedAdapter(
            api_key=preset.api_key or "ollama",
            base_url=preset.base_url,
        )

    raise ValueError(f"No embed adapter wired for provider: {name!r}")
