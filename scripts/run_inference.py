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

    request = prepare_chat_request(
        ChatCompletionRequest(messages=[ChatMessage(role="user", content=prompt)]),
        provider=provider_name,
    )
    provider = get_capability(Capability.CHAT)(provider_name)

    if use_rag:
        from app.rag.pipeline import default_pipeline
        if stream:
            logging.info("RAG + Streaming")
            chunks = None
            async for delta, retrieved in default_pipeline.answer_stream(prompt, provider_name=provider_name):
                chunks = retrieved
                print(delta, end="", flush=True)
            print()
            if chunks:
                print("\n--- sources ---")
                for c in chunks:
                    print(f"  [{c.score:.2f}] {c.source}: {c.text[:80]}...")
        else:
            logging.info("RAG & No streaming")
            result = await default_pipeline.answer(prompt, provider_name=provider_name)
            print(result.text)
            if result.sources:
                print("\n--- sources ---")
                for c in result.sources:
                    print(f"  [{c.score:.2f}] {c.source}: {c.text[:80]}...")
    else:
        if stream:
            logging.info("No RAG + Streaming")
            async for delta in provider.stream(request):
                print(delta, end="", flush=True)
            print()
        else:
            logging.info("No RAG & No streaming")
            response = await provider.complete(request)
            print(response.text)


if __name__ == "__main__":
    asyncio.run(main())