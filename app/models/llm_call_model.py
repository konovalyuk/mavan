from dataclasses import dataclass


@dataclass
class LLMCallRecord:
    capability: str          # chat | embed | rerank | agent
    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    user_id: str | None
    chat_id: str | None
    tool_calls_count: int = 0