from app.models.data_models import ChatMessage
from config import llm_settings
from app.llm.chat.providers.anthropic import AnthropicChatAdapter
from app.llm.chat.providers.mock import MockChatAdapter
from app.llm.chat.providers.openai_compatible import OpenAICompatibleChatAdapter
from app.llm.chat.types import ChatProvider, ChatRequest

CHAT_PROVIDERS = frozenset({
    "mock", "openai", "gemini", "mistral", "deepseek",
    "huggingface", "github_models", "ollama", "anthropic",
})

OPENAI_COMPATIBLE_CHAT = frozenset({
    "openai", "gemini", "mistral", "deepseek",
    "huggingface", "github_models", "ollama",
})


def chat_provider(provider: str | None = None) -> str:
    name = (provider or llm_settings.CHAT_PROVIDER).lower()
    if name not in CHAT_PROVIDERS:
        raise ValueError(f"Provider {name!r} does not support chat. " f"Supported: {', '.join(sorted(CHAT_PROVIDERS))}")
    return name


def get_chat_provider(provider: str | None = None) -> ChatProvider:
    name = chat_provider(provider)

    if name == "mock":
        return MockChatAdapter()
    preset = llm_settings.provider_presets()[name]
    if name in OPENAI_COMPATIBLE_CHAT:
        if not preset.api_key and name != "ollama":
            raise ValueError(f"API key missing for LLM provider: {name}")
        return OpenAICompatibleChatAdapter(
            api_key=preset.api_key or "ollama",
            base_url=preset.base_url,
            default_model=preset.default_model,
        )
    if name == "anthropic":
        if not preset.api_key:
            raise ValueError(f"API key missing for LLM provider: {name}")
        return AnthropicChatAdapter(
            api_key=preset.api_key,
            base_url=preset.base_url,
            default_model=preset.default_model,
        )
    # if p == "llama_local":
    #     from app.llm.llama_local_provider import LlamaLocalProvider
    #     return LlamaLocalProvider()
    raise ValueError(f"No chat adapter wired for provider: {name!r}")


def resolve_chat_model(*, request_model: str | None = None, provider: str | None = None) -> str:
    if request_model is not None:
        return request_model
    if llm_settings.CHAT_MODEL:
        return llm_settings.CHAT_MODEL

    name = chat_provider(provider)
    if name == "mock":
        return llm_settings.CHAT_MODEL  # или любая заглушка

    return llm_settings.provider_presets()[name].default_model


def build_chat_request(
        *,
        messages: list[ChatMessage],
        request_model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        provider: str | None = None,
) -> ChatRequest:
    return ChatRequest(
        messages=messages,
        model=resolve_chat_model(request_model=request_model, provider=provider),
        max_tokens=max_tokens if max_tokens is not None else llm_settings.DEFAULT_MAX_TOKENS,
        temperature=temperature if temperature is not None else llm_settings.DEFAULT_TEMPERATURE,
    )
