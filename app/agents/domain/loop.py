import json


async def llm_json(prompt: str, *, provider: str | None = None) -> dict | list:
    """Small helper for structured LLM JSON used by extract (not an agent)."""
    from app.llm.capabilities import Capability, get_capability
    from app.llm.chat.chat_providers import prepare_chat_request
    from app.llm.chat.schemas import ChatCompletionRequest, ChatMessage

    chat = get_capability(Capability.CHAT)(provider)
    req = prepare_chat_request(
        ChatCompletionRequest(messages=[ChatMessage(role="user", content=prompt)], temperature=0.1),
        provider_name=provider,
    )
    resp = await chat.complete(req)
    text = (resp.text or "").strip()
    if "```" in text:
        text = text.split("```")[1].replace("json", "", 1).strip()
    return json.loads(text)
