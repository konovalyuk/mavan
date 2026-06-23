from typing import AsyncIterator, Protocol
from app.models.data_models import ChatMessage


class LLMProvider(Protocol):
    async def stream(
            self,
            *,
            messages: list[ChatMessage],
            model: str | None,
            max_tokens: int | None,
            temperature: float | None,
    ) -> AsyncIterator[str]:
        """Yields text deltas (pieces), not accumulated text."""

