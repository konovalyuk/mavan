from datetime import datetime, timezone
from app.agents.loop import run_tool_loop
from app.database import get_db


async def run_agent(question: str, *, provider: str | None = None) -> dict:
    answer, tool_log = await run_tool_loop(question, provider=provider)
    await get_db().agent_traces.insert_one({
        "question": question,
        "tool_log": tool_log,
        "created_at": datetime.now(timezone.utc),
    })
    return {
        "answer": answer,
        "agent": "rag",
        "tool_log": tool_log,
    }
