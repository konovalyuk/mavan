from app.agents.types import AgentRuntime
from app.agents.adk.search_agent import adk_agent

AGENT_BACKENDS = frozenset({"adk"})
_AGENTS: dict[str, AgentRuntime] = {
    "adk": adk_agent,
}


def get_runtime(backend: str) -> AgentRuntime:
    try:
        return _AGENTS[backend]
    except KeyError:
        supported = ", ".join(sorted(_AGENTS))
        raise ValueError(f"Unknown agent backend {backend!r}. Supported: {supported}") from None
