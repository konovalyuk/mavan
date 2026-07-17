#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: F401

from app.agents.registry import get_runtime

async def main():
    args = sys.argv[1:]
    session_id = None
    if args and args[0].startswith("--session="):
        session_id = args.pop(0).split("=", 1)[1]

    prompt = " ".join(args) if args else input("You: ")
    result = await get_runtime("adk").run(prompt, session_id=session_id)

    print(result.answer)
    if result.sources:
        print("\n--- sources ---")
        for s in result.sources:
            print(f"  [{s['title']}] {s['url']}")

if __name__ == "__main__":
    asyncio.run(main())