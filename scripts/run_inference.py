#!/usr/bin/env python3
import asyncio
import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.llm.capabilities import Capability, get_capability
from app.llm.chat.chat_providers import build_chat_request, CHAT_PROVIDERS
from app.models.data_models import ChatMessage

logging.basicConfig(level=logging.INFO)

async def main():
    args = sys.argv[1:]
    stream = bool(args and args[0] == "--stream" and args.pop(0))

    provider_name = None
    if args and not args[0].startswith("-") and args[0] in CHAT_PROVIDERS:
        provider_name = args.pop(0)

    prompt = " ".join(args) if args else input("You: ")

    request = build_chat_request(
        messages=[ChatMessage(role="user", content=prompt)],
        provider=provider_name,
    )
    provider = get_capability(Capability.CHAT)(provider_name)

    if stream:
        async for delta in provider.stream(request):
            print(delta, end="", flush=True)
        print()
    else:
        response = await provider.complete(request)
        print(response.text)


if __name__ == "__main__":
    asyncio.run(main())