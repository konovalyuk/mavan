#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agents.tool_loop.loop import run_tool_loop, resolve_tools
from app.agents.types import AGENT_TEMPERATURE, AGENT_TOOL_CHOICE
from app.llm.capabilities import Capability, get_capability
from app.llm.chat.chat_providers import CHAT_PROVIDERS, prepare_chat_request
from app.llm.chat.schemas import ChatCompletionRequest, ChatMessage
from app.services.prompts import build_system_content


async def main():
    args = sys.argv[1:]
    provider = None
    if args and args[0] in CHAT_PROVIDERS:
        provider = args.pop(0)
    question = " ".join(args) if args else input("You: ")

    chat_provider = get_capability(Capability.CHAT)(provider)
    request = prepare_chat_request(
        ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content=build_system_content(mode="agent")),
                ChatMessage(role="user", content=question),
            ],
            tools=resolve_tools("default"),
            tool_choice=AGENT_TOOL_CHOICE,
            temperature=AGENT_TEMPERATURE,
        ),
        provider_name=provider,
    )
    result = await run_tool_loop(request=request, chat_provider=chat_provider)
    print(result.answer)
    if result.tool_log:
        print("\n--- tool calls ---")
        for entry in result.tool_log:
            print(f"  {entry['tool']}({entry['args']})")


if __name__ == "__main__":
    asyncio.run(main())
