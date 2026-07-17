import json
import logging
from dataclasses import dataclass
from fastapi import HTTPException

from app.agents.tool_loop.tools import (
    CALCULATOR_TOOL,
    SEARCH_NOTES_TOOL,
    WEB_SEARCH_TOOL,
    execute_tool,
)
from app.llm.chat.schemas import ChatCompletionRequest, ChatMessage
from app.llm.chat.types import ChatProvider

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 5

TOOL_PRESETS: dict[str, list[dict]] = {
    "notes": [SEARCH_NOTES_TOOL],
    "web": [WEB_SEARCH_TOOL],
    "python": [CALCULATOR_TOOL],
    "default": [SEARCH_NOTES_TOOL],
    "full": [SEARCH_NOTES_TOOL, WEB_SEARCH_TOOL, CALCULATOR_TOOL],
}


@dataclass(frozen=True)
class ToolLoopResult:
    answer: str
    tool_log: list[dict]


def resolve_tools(agent_tools: str | list[str] | None) -> list[dict]:
    if agent_tools is None:
        return TOOL_PRESETS["default"]
    if isinstance(agent_tools, str):
        if agent_tools not in TOOL_PRESETS:
            raise HTTPException(status_code=400, detail=f"Unknown agent_tools preset {agent_tools!r}. Supported: {sorted(TOOL_PRESETS)}")
        return TOOL_PRESETS[agent_tools]
    tools: list[dict] = []
    for name in agent_tools:
        if name not in TOOL_PRESETS:
            raise HTTPException(status_code=400, detail=f"Unknown tool {name!r} in agent_tools")
        tools.extend(TOOL_PRESETS[name])
    return tools


async def run_tool_loop(request: ChatCompletionRequest, chat_provider: ChatProvider, source_prefixes: tuple[str, ...] | None = None) -> ToolLoopResult:
    tool_log: list[dict] = []

    for _ in range(MAX_TOOL_ROUNDS):
        response = await chat_provider.complete(request)

        if not response.tool_calls:
            return ToolLoopResult(answer=response.text or "", tool_log=tool_log)

        request.messages.append(ChatMessage(
            role="assistant",
            content=response.text,
            tool_calls=response.tool_calls,
        ))

        for tc in response.tool_calls:
            fn = tc["function"]
            args = json.loads(fn["arguments"])
            result, meta = await execute_tool(fn["name"], args, source_prefixes=source_prefixes)
            tool_log.append({"tool": fn["name"], "args": args, "meta": meta})
            request.messages.append(ChatMessage(
                role="tool",
                tool_call_id=tc["id"],
                content=result,
            ))

    return ToolLoopResult(answer="Agent stopped: max tool rounds reached.", tool_log=tool_log)
