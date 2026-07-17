import asyncio
import json

from app.decision.fuse import average_llm, merge_forecasts
from app.domain import store
from config import domain_settings
from app.domain_model.inference import predict
from app.llm.capabilities import Capability, get_capability
from app.llm.chat.chat_providers import prepare_chat_request
from app.llm.chat.schemas import ChatCompletionRequest, ChatMessage

FORECAST_PROMPT = """Domain: {name}
Description: {description}
Known terminal outcome states: {known_states}
Context state: {context_state}
Candidate actions: {actions}

For EACH action estimate probable FINAL outcome states (not intermediate steps), sum ~1.0 per action.
Return ONLY JSON: {{"Action name": {{"Outcome state": 0.0-1.0}}}}
"""


def _parse_json(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1].replace("json", "", 1).strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict[str, float]] = {}
    for action, states in data.items():
        if isinstance(states, dict):
            out[str(action)] = {str(st): float(p) for st, p in states.items()}
    return out


async def _forecast_one(provider: str, prompt: str) -> dict[str, dict[str, float]]:
    chat = get_capability(Capability.CHAT)(provider.strip())
    req = prepare_chat_request(
        ChatCompletionRequest(messages=[ChatMessage(role="user", content=prompt)], temperature=0.2),
        provider_name=provider.strip(),
    )
    resp = await chat.complete(req)
    try:
        return _parse_json(resp.text or "")
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


async def _core_states(domain_id: str, limit: int = 50) -> list[str]:
    seen: set[str] = set()
    for doc in await store.get_core_memory(domain_id):
        seen.add(doc.get("state", ""))
        for o in doc.get("outcome_states") or []:
            seen.add(o.get("state", ""))
    return sorted(s for s in seen if s)[:limit]


async def run_forecast(
    domain_id: str, context_state: str, candidate_actions: list[str], *, provider: str | None = None,
) -> dict:
    domain = await store.get_domain(domain_id) or {}
    known = await _core_states(domain_id)
    prior = {a: predict(domain_id, context_state, a) for a in candidate_actions}

    prompt = FORECAST_PROMPT.format(
        name=domain.get("name", domain_id),
        description=domain.get("description", ""),
        known_states=json.dumps(known, ensure_ascii=False),
        context_state=context_state,
        actions=json.dumps(candidate_actions, ensure_ascii=False),
    )
    providers = [p.strip() for p in domain_settings.FORECAST_PROVIDERS.split(",") if p.strip()]
    if not providers:
        providers = [provider or "mock"]

    try:
        raw = await asyncio.wait_for(
            asyncio.gather(*[_forecast_one(p, prompt) for p in providers], return_exceptions=True),
            timeout=domain_settings.FORECAST_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        raw = []

    llm_results = [r for r in raw if isinstance(r, dict) and r]
    llm_avg = average_llm(llm_results) if llm_results else {a: {"unknown": 1.0} for a in candidate_actions}
    return merge_forecasts(prior, llm_avg)
