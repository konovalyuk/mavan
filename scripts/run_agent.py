#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agents.rag_agent import run_rag_agent
from app.llm.chat.chat_providers import CHAT_PROVIDERS


async def main():
    args = sys.argv[1:]
    provider = None
    if args and args[0] in CHAT_PROVIDERS:
        provider = args.pop(0)
    question = " ".join(args) if args else input("You: ")
    answer, sources = await run_rag_agent(question, provider=provider)
    print(answer)
    if sources:
        print("\n--- sources ---")
        for c in sources:
            print(f"  [{c.score:.2f}] {c.source}: {c.text[:80]}...")


if __name__ == "__main__":
    asyncio.run(main())
