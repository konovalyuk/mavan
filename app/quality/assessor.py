import asyncio
import json

from config import domain_settings
from app.domain.schemas import QualityScores
from app.llm.capabilities import Capability, get_capability
from app.llm.chat.chat_providers import prepare_chat_request
from app.llm.chat.schemas import ChatCompletionRequest, ChatMessage

QUALITY_PROMPT = """Assess this text for decision intelligence training. Return ONLY JSON:
{{"credibility": 0-100, "completeness": 0-100, "depth": 0-100, "terminology": 0-100, "coherence": 0-100, "misunderstanding_risk": 0-100}}

Text:
{text}
"""


async def _score_one(provider: str, text: str) -> dict:
    chat = get_capability(Capability.CHAT)(provider.strip())
    req = prepare_chat_request(
        ChatCompletionRequest(
            messages=[ChatMessage(role="user", content=QUALITY_PROMPT.format(text=text[:8000]))],
            temperature=0.0,
        ),
        provider_name=provider.strip(),
    )
    resp = await chat.complete(req)
    raw = (resp.text or "").strip()
    if "```" in raw:
        raw = raw.split("```")[1].replace("json", "", 1).strip()
    data = json.loads(raw)
    return {"provider": provider.strip(), "scores": data}


def _mean_scores(results: list[dict]) -> QualityScores:
    keys = ["credibility", "completeness", "depth", "terminology", "coherence", "misunderstanding_risk"]
    avg = {}
    for k in keys:
        vals = [r["scores"].get(k, 0) for r in results if "scores" in r]
        avg[k] = sum(vals) / len(vals) if vals else 0.0
    return QualityScores(**avg)


def _passes(scores: QualityScores) -> bool:
    s = domain_settings
    return (
        scores.credibility >= s.QUALITY_CREDIBILITY_MIN
        and scores.completeness >= s.QUALITY_COMPLETENESS_MIN
        and scores.depth >= s.QUALITY_DEPTH_MIN
        and scores.terminology >= s.QUALITY_TERMINOLOGY_MIN
        and scores.coherence >= s.QUALITY_COHERENCE_MIN
        and scores.misunderstanding_risk <= s.QUALITY_MISUNDERSTANDING_MAX
    )


async def assess(text: str, *, provider: str | None = None) -> dict:
    providers = [p.strip() for p in domain_settings.QUALITY_PROVIDERS.split(",") if p.strip()]
    if not providers:
        providers = [provider or "mock"]

    async def run():
        return await asyncio.gather(*[_score_one(p, text) for p in providers], return_exceptions=True)

    try:
        raw = await asyncio.wait_for(run(), timeout=domain_settings.QUALITY_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        raw = []

    results = [r for r in raw if isinstance(r, dict)]
    if not results:
        results = [{"provider": "fallback", "scores": {
            "credibility": 50, "completeness": 50, "depth": 50,
            "terminology": 50, "coherence": 50, "misunderstanding_risk": 50,
        }}]

    avg = _mean_scores(results)
    passed = _passes(avg)
    return {
        "scores": avg.model_dump(),
        "avg": avg.avg(),
        "passed": passed,
        "per_provider": results,
    }
