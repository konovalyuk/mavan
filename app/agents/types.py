from dataclasses import dataclass, field
from typing import Protocol

AGENT_TEMPERATURE = 0.2
AGENT_TOOL_CHOICE = "auto"

@dataclass(frozen=True)
class AgentResult:
    answer: str
    backend: str
    tool_log: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)  # ADK grounding
    session_id: str | None = None


class AgentRuntime(Protocol):
    async def run(self, question: str, *, provider: str | None = None, session_id: str | None = None) -> AgentResult: ...
