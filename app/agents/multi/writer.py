from app.llm.capabilities import Capability, get_capability
from app.llm.chat.chat_providers import prepare_chat_request
from app.llm.chat.schemas import ChatCompletionRequest, ChatMessage

WRITER_SYSTEM = "You are a writer agent. Use only the research notes below."


async def run_writer(question: str, research_text: str, *, provider: str | None = None) -> str:
    messages = [
        ChatMessage(role="system", content=WRITER_SYSTEM),
        ChatMessage(role="user", content=f"Question:\n{question}\n\nResearch:\n{research_text}"),
    ]
    chat = get_capability(Capability.CHAT)(provider)
    response = await chat.complete(
        prepare_chat_request(ChatCompletionRequest(messages=messages, temperature=0.3), provider=provider)
    )
    return response.text or ""
