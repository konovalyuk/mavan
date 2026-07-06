# app/services/llm_complete_service.py
import time
from app.llm.chat.types import ChatProvider
from app.llm.chat.schemas import ChatCompletionRequest, ChatCompletionResponse
from app.models.llm_call_model import LLMCallRecord
from app.services.monitoring.llm_metrics_service import record_llm_call


async def complete_and_record(
        provider: ChatProvider,
        request: ChatCompletionRequest,
        *,
        capability: str,
        provider_name: str | None,
        user_id: str | None = None,
        chat_id: str | None = None,
        tool_calls_count: int = 0,
) -> ChatCompletionResponse:
    t0 = time.perf_counter()
    response = await provider.complete(request)
    latency_ms = (time.perf_counter() - t0) * 1000

    usage = response.usage or {}
    await record_llm_call(LLMCallRecord(
        capability=capability,
        provider=provider_name or "unknown",
        model=response.model or request.model or "unknown",
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        latency_ms=latency_ms,
        user_id=user_id,
        chat_id=chat_id,
        tool_calls_count=tool_calls_count,
    ))
    return response
