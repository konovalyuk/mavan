import json
from app.agents.tools import SEARCH_NOTES_TOOL, execute_tool
from app.llm.capabilities import Capability, get_capability
from app.llm.chat.chat_providers import prepare_chat_request
from app.llm.chat.schemas import ChatCompletionRequest, ChatMessage

AGENT_SYSTEM = (
    "You are a helpful assistant with access to search_notes.\n"
    "Call search_notes when you need facts from project notes.\n"
    "You may call it multiple times with different queries.\n"
    "If notes don't help, say you don't know."
)

MAX_TOOL_ROUNDS = 5


async def run_tool_loop(question: str, *, provider: str | None = None, tools: list[dict] | None = None,
                        system: str | None = None) -> tuple[str, list[dict]]:
    tools = tools or [SEARCH_NOTES_TOOL]
    messages = [
        ChatMessage(role="system", content=system or AGENT_SYSTEM),
        ChatMessage(role="user", content=question),
    ]
    chat = get_capability(Capability.CHAT)(provider)
    tool_log: list[dict] = []

    for _ in range(MAX_TOOL_ROUNDS):
        request = prepare_chat_request(
            ChatCompletionRequest(messages=messages, tools=tools, tool_choice="auto", temperature=0.2),
            provider=provider,
        )
        response = await chat.complete(request)

        if not response.tool_calls:
            return response.text or "", tool_log

        # assistant message with tool_calls must go back into history
        messages.append(ChatMessage(
            role="assistant",
            content=response.text,
            tool_calls=response.tool_calls,
        ))

        for tc in response.tool_calls:
            fn = tc["function"]
            args = json.loads(fn["arguments"])
            result, meta = await execute_tool(fn["name"], args)
            tool_log.append({"tool": fn["name"], "args": args, "meta": meta})
            messages.append(ChatMessage(
                role="tool",
                tool_call_id=tc["id"],
                content=result,
            ))

    return "Agent stopped: max tool rounds reached.", tool_log
