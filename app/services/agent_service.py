from app.agents.rag_agent import run_rag_agent


async def run_agent(question: str, *, provider: str | None = None) -> dict:
    answer, sources = await run_rag_agent(question, provider=provider)
    return {
        "answer": answer,
        "agent": "rag",
        "sources": [
            {"text": c.text, "score": c.score, "source": c.source}
            for c in sources
        ],
    }