#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agents.multi.orchestrator import run_multi_agent
from app.llm.chat.chat_providers import CHAT_PROVIDERS


async def main():
    args = sys.argv[1:]
    provider = None
    if args and args[0] in CHAT_PROVIDERS:
        provider = args.pop(0)

    question = " ".join(args) if args else input("You: ")
    result = await run_multi_agent(question, provider=provider)

    print("\n=== ANSWER ===")
    print(result["answer"])

    print("\n=== RESEARCH (intermediate) ===")
    print(result["research"])

    if result["tool_log"]:
        print("\n=== TOOL CALLS ===")
        for entry in result["tool_log"]:
            print(f"  {entry['tool']}({entry['args']})")


if __name__ == "__main__":
    asyncio.run(main())
