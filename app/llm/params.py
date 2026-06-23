from dataclasses import dataclass

from app.models.data_models import ChatCompletionRequest
from config import llm_settings


@dataclass(frozen=True)
class CompletionContext:
    """Resolved parameters for one completion request."""
    model: str
    max_tokens: int
    temperature: float

    @classmethod
    def from_request(
        cls,
        request: ChatCompletionRequest,
        *,
        chat_record: dict | None = None,
    ) -> "CompletionContext":
        # Priority: request → existing chat → env default
        model = request.model
        if model is None and chat_record:
            model = chat_record.get("model")
        if model is None:
            model = llm_settings.MODEL

        return cls(
            model=model,
            max_tokens=(
                request.max_tokens
                if request.max_tokens is not None
                else llm_settings.DEFAULT_MAX_TOKENS
            ),
            temperature=(
                request.temperature
                if request.temperature is not None
                else llm_settings.DEFAULT_TEMPERATURE
            ),
        )

    def as_llm_kwargs(self) -> dict:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }


# backward compat
def resolve_llm_params(request: ChatCompletionRequest, chat_record: dict | None = None) -> dict:
    return CompletionContext.from_request(request, chat_record=chat_record).as_llm_kwargs()