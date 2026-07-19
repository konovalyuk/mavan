from app.agents.adk.helpers import extract_final_text, extract_grounding_sources
from config import llm_settings
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

search_agent = Agent(
    name="helpful_assistant",
    model=Gemini(model=llm_settings.GEMINI_MODEL, retry_options=retry_config),
    description="Web search assistant.",
    instruction="You are a helpful assistant. Use Google Search for current info or if unsure.",
    tools=[google_search],
)

google_search_agent_runner = InMemoryRunner(agent=search_agent)


async def adk_search_agent(question: str, session_id: str | None = None) -> tuple[str, list[dict]]:
    sid = session_id or "mavan_adk"
    events = await google_search_agent_runner.run_debug(question, quiet=True, user_id="mavan", session_id=sid)
    return extract_final_text(events), extract_grounding_sources(events)
