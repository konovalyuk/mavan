from app.agents.loop import run_tool_loop

RESEARCH_SYSTEM = (
    "You are a research agent. Use search_notes to gather facts.\n"
    "Return a concise research summary with source references."
)

async def run_researcher(question: str, *, provider: str | None = None) -> tuple[str, list[dict]]:
    # run_tool_loop с кастомным system prompt — добавьте параметр system= в loop.py
    return await run_tool_loop(question, provider=provider, system=RESEARCH_SYSTEM)