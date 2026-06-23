import asyncio
import random


class MockLLMProvider:
    async def stream(self, *, messages, model=None, max_tokens=None, temperature=None):
        user = next((m.content for m in messages if m.role == "user"), "")
        words = (user or "Hello from mock LLM").split()
        reply = " ".join(random.sample(words, min(len(words), 5)) if words else ["Mock", "response", "works."])
        for word in reply.split():
            yield word + " "
            await asyncio.sleep(0.08)  # имитация latency
