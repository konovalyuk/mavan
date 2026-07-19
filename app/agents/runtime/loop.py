from __future__ import annotations

import json
import logging

from app.agents.runtime.types import AgentLoopResult, Agent, ToolExecutor, resolve_provider
from config import agent_settings
from app.llm.capabilities import Capability, get_capability
from app.llm.chat.chat_providers import prepare_chat_request
from app.llm.chat.schemas import ChatCompletionRequest, ChatMessage
from app.llm.chat.types import ChatProvider

logger = logging.getLogger(__name__)


async def run_agent_loop(request: ChatCompletionRequest, chat_provider: ChatProvider, execute: ToolExecutor, max_rounds: int) -> AgentLoopResult:
    """Observe → reason → act until the model stops calling tools or max_rounds."""
    tool_log: list[dict] = []
    rounds = 0

    for rounds in range(1, max_rounds + 1):
        response = await chat_provider.complete(request)

        if not response.tool_calls:
            return AgentLoopResult(answer=response.text or "", tool_log=tool_log, rounds=rounds)

        request.messages.append(ChatMessage(role="assistant", content=response.text, tool_calls=response.tool_calls))

        for tc in response.tool_calls:
            fn = tc["function"]
            name = fn["name"]
            try:
                args = json.loads(fn["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
                result, meta = f"invalid JSON arguments for {name}", {"error": "bad_args"}
            else:
                try:
                    result, meta = await execute(name, args)
                except Exception as e:
                    logger.exception("tool %s failed", name)
                    result, meta = f"tool error: {e}", {"error": str(e)}

            tool_log.append({"tool": name, "args": args, "meta": meta})
            request.messages.append(ChatMessage(
                role="tool",
                tool_call_id=tc["id"],
                content=result if isinstance(result, str) else json.dumps(result, default=str),
            ))

    return AgentLoopResult(answer="Agent stopped: max tool rounds reached.", tool_log=tool_log, rounds=max_rounds)


async def run_agent(agent: Agent, user_message: str, execute: ToolExecutor | None = None) -> AgentLoopResult:
    """Run one real LlmAgent using its own provider/model."""
    provider = resolve_provider(agent.provider)
    system = (
        f"You are {agent.name}.\n"
        f"Goal: {agent.goal}\n"
        f"{agent.instruction}\n"
        "Use tools to act when available. When the goal is met (or you cannot progress), "
        "stop calling tools and reply with a short status summary."
    )
    tools = agent.tools or None
    chat_provider = get_capability(Capability.CHAT)(provider)
    request = prepare_chat_request(
        ChatCompletionRequest(
            messages=[ChatMessage(role="system", content=system), ChatMessage(role="user", content=user_message)],
            model=agent.model,
            tools=tools,
            tool_choice=agent_settings.AGENT_TOOL_CHOICE if tools else None,
            temperature=agent_settings.AGENT_TEMPERATURE,
        ),
        provider_name=provider,
    )

    async def _noop_execute(name: str, args: dict) -> tuple[str, dict]:
        raise ValueError(f"No tools registered but model called {name}")

    return await run_agent_loop(request, chat_provider, execute or _noop_execute, max_rounds=agent.max_rounds if tools else 1)
