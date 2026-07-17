#!/usr/bin/env python3
import asyncio
import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.llm.capabilities import Capability, get_capability
from app.llm.chat.chat_providers import CHAT_PROVIDERS, prepare_chat_request
from app.llm.chat.schemas import ChatCompletionRequest, ChatMessage
from app.rag.pipeline import get_default_pipeline
from app.services.prompts import build_system_content

logging.basicConfig(level=logging.INFO)


async def main():
    args = sys.argv[1:]
    use_rag = True
    if args and args[0] == "--no-rag":
        use_rag = False
        args.pop(0)

    stream = bool(args and args[0] == "--stream" and args.pop(0))

    provider_name = None
    if args and not args[0].startswith("-") and args[0] in CHAT_PROVIDERS:
        provider_name = args.pop(0)

    prompt = " ".join(args) if args else input("You: ")
    provider = get_capability(Capability.CHAT)(provider_name)
    chunks = []

    if use_rag:
        chunks = await get_default_pipeline().retrieve_chunks(prompt)
        messages = [
            ChatMessage(role="system", content=build_system_content(chunks=chunks)),
            ChatMessage(role="user", content=prompt),
        ]
        logging.info("RAG + %s", "Streaming" if stream else "No streaming")
    else:
        messages = [ChatMessage(role="user", content=prompt)]
        logging.info("No RAG + %s", "Streaming" if stream else "No streaming")

    request = prepare_chat_request(
        ChatCompletionRequest(messages=messages),
        provider_name=provider_name,
    )

    if stream:
        async for delta in provider.stream(request):
            print(delta, end="", flush=True)
        print()
    else:
        response = await provider.complete(request)
        print(response.text)

    if use_rag and chunks:
        print("\n--- sources ---")
        for c in chunks:
            print(f"  [{c.score:.2f}] {c.source}: {c.text[:80]}...")


if __name__ == "__main__":
    asyncio.run(main())
