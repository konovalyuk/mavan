import json

from app.agents.domain.tools import google_search
from app.domain import store

DISCOVERY_PROMPT = """Domain: {name}
Description: {description}

Return JSON list of 3-5 Google search queries to find historical information sources for this domain.
Example: ["query 1", "query 2"]
"""


async def run_discovery(domain_id: str, *, provider: str | None = None) -> int:
    domain = await store.get_domain(domain_id)
    if not domain:
        raise ValueError("domain not found")
    queries = await llm_json(
        DISCOVERY_PROMPT.format(name=domain["name"], description=domain.get("description", "")),
        provider=provider,
    )
    if isinstance(queries, dict):
        queries = queries.get("queries", [])
    if not isinstance(queries, list):
        queries = [str(queries)]

    seen, urls = set(), []
    for q in queries[:5]:
        result, _ = await google_search(str(q))
        data = json.loads(result)
        for r in data.get("results", []):
            url = r.get("url", "")
            if url and url not in seen:
                seen.add(url)
                urls.append({"url": url, "title": r.get("title", "")})
    return await store.upsert_sources(domain_id, urls)


async def llm_json(prompt: str, *, provider: str | None = None) -> dict | list:
    from app.llm.capabilities import Capability, get_capability
    from app.llm.chat.chat_providers import prepare_chat_request
    from app.llm.chat.schemas import ChatCompletionRequest, ChatMessage

    chat = get_capability(Capability.CHAT)(provider)
    req = prepare_chat_request(
        ChatCompletionRequest(messages=[ChatMessage(role="user", content=prompt)], temperature=0.1),
        provider=provider,
    )
    resp = await chat.complete(req)
    text = (resp.text or "").strip()
    if "```" in text:
        text = text.split("```")[1].replace("json", "", 1).strip()
    return json.loads(text)
