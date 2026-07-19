from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from config import llm_settings

ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[tuple[str, dict[str, Any]]]]


def resolve_provider(*candidates: str | None) -> str:
    for c in candidates:
        if c and str(c).strip():
            return str(c).strip()
    return (llm_settings.CHAT_PROVIDER or "mock").strip()


@dataclass(frozen=True)
class Agent:
    """Real agent: LLM + model/provider + goal + tools + observe→reason→act loop."""

    name: str
    goal: str
    instruction: str
    tools: list[dict[str, Any]]
    provider: str
    model: str | None = None
    max_rounds: int = 6


@dataclass
class SessionState:
    """Shared packet state for domain learning (not an agent)."""

    domain_id: str = ""
    domain_name: str = ""
    domain_description: str = ""
    provider_override: str | None = None
    packet_text: str = ""
    packet_urls: list[str] = field(default_factory=list)
    feedback: str = ""
    quality: dict[str, Any] = field(default_factory=dict)
    sao: list[dict] = field(default_factory=list)
    train_result: dict[str, Any] = field(default_factory=dict)
    packet_done: bool = False
    seen_urls: set[str] = field(default_factory=set)
    metrics: dict[str, int] = field(default_factory=lambda: {
        "research_calls": 0,
        "quality_pass": 0,
        "quality_fail": 0,
        "extract_ok": 0,
        "extract_fail": 0,
        "train_ok": 0,
        "train_fail": 0,
        "coordinator_rounds": 0,
    })


@dataclass(frozen=True)
class AgentLoopResult:
    answer: str
    tool_log: list[dict[str, Any]] = field(default_factory=list)
    rounds: int = 0
