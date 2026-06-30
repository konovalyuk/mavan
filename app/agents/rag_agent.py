from app.agents.tools import search_notes
from app.llm.capabilities import Capability, get_capability
from app.llm.chat.chat_providers import prepare_chat_request
from app.llm.chat.schemas import ChatCompletionRequest, ChatMessage

AGENT_SYSTEM = (
    "You are a helpful assistant with access to project notes.\n"
    "A search tool was already run for you. Use only the tool result below.\n"
    "If the tool found nothing useful, say you don't know."
)


async def run_rag_agent(question: str, *, provider: str | None = None) -> str:
    """
    Простой агент в 2 шага:
    1) tool search_notes  (RAG)
    2) chat формирует финальный ответ
    """
    tool_result = await search_notes(question, provider=provider)

    messages = [
        ChatMessage(role="system", content=AGENT_SYSTEM),
        ChatMessage(role="user", content=question),
        ChatMessage(
            role="assistant",
            content=f"[Tool: search_notes]\n{tool_result}",
        ),
        ChatMessage(
            role="user",
            content="Using the tool result above, answer the question.",
        ),
    ]

    request = prepare_chat_request(
        ChatCompletionRequest(messages=messages, temperature=0.2),
        provider=provider,
    )
    chat = get_capability(Capability.CHAT)(provider)
    response = await chat.complete(request)
    return response.text
