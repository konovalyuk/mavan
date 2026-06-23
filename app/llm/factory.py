from config import llm_settings
from app.llm.mock_provider import MockLLMProvider
from app.llm.open_ai_compatible_provider import OpenAICompatibleProvider


def get_llm_provider():
    p = llm_settings.PROVIDER.lower()
    if p == "mock":
        return MockLLMProvider()
    if p == "openai":
        return OpenAICompatibleProvider(
            base_url=llm_settings.OPENAI_BASE_URL or None,
            api_key=llm_settings.OPENAI_API_KEY,
        )
    if p == "ollama":
        return OpenAICompatibleProvider(
            base_url=llm_settings.OLLAMA_BASE_URL,
            api_key="ollama",
        )
    # if p == "gemini":
    #     from app.llm.gemini_provider import GeminiProvider
    #     return GeminiProvider()
    raise ValueError(f"Unknown LLM_PROVIDER: {p}")
