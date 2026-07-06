from app.agents.multi.researcher import run_researcher
from app.agents.multi.writer import run_writer


async def run_multi_agent(question: str, *, provider: str | None = None) -> dict:
    research_text, tool_log = await run_researcher(question, provider=provider)
    final = await run_writer(question, research_text, provider=provider)
    return {
        "answer": final,
        "research": research_text,
        "tool_log": tool_log,
        "agent": "multi-rag",
    }
