"""Data quality: LlmAgent judges (per provider) + code aggregation."""
from __future__ import annotations

import asyncio
import json
import logging

from app.agents.runtime import Agent, run_agent
from app.domain.schemas import QualityScores
from config import domain_settings, llm_settings

logger = logging.getLogger(__name__)

_KEYS = ("credibility", "completeness", "depth", "terminology", "coherence", "misunderstanding_risk")

QUALITY_INSTRUCTION = (
    "Assess text for decision-intelligence training. Return ONLY JSON:\n"
    '{"credibility":0-100,"completeness":0-100,"depth":0-100,'
    '"terminology":0-100,"coherence":0-100,"misunderstanding_risk":0-100}'
)


def mean_scores(results: list[dict]) -> QualityScores:
    avg = {
        k: (sum(r["scores"].get(k, 0) for r in results if "scores" in r)
            / max(1, sum(1 for r in results if "scores" in r)))
        for k in _KEYS
    }
    return QualityScores(**avg)


def passes(scores: QualityScores) -> bool:
    s = domain_settings
    return (
        scores.credibility >= s.QUALITY_CREDIBILITY_MIN
        and scores.completeness >= s.QUALITY_COMPLETENESS_MIN
        and scores.depth >= s.QUALITY_DEPTH_MIN
        and scores.terminology >= s.QUALITY_TERMINOLOGY_MIN
        and scores.coherence >= s.QUALITY_COHERENCE_MIN
        and scores.misunderstanding_risk <= s.QUALITY_MISUNDERSTANDING_MAX
    )


def _parse_scores(text: str) -> dict:
    raw = (text or "").strip()
    if "```" in raw:
        raw = raw.split("```")[1].replace("json", "", 1).strip()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("scores not a dict")
    return data


async def _judge(provider: str, text: str) -> dict:
    agent = Agent(
        name=f"QualityJudge_{provider}",
        goal="Score text quality as JSON only.",
        instruction=QUALITY_INSTRUCTION,
        tools=[],
        provider=provider.strip(),
        max_rounds=1,
    )
    result = await run_agent(agent, text[:8000])
    try:
        return {"provider": provider.strip(), "scores": _parse_scores(result.answer)}
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("quality judge %s failed: %s", provider, e)
        return {"provider": provider.strip(), "error": str(e)}


def aggregate_quality(results: list[dict]) -> dict:
    ok = [r for r in results if "scores" in r]
    if not ok:
        return {
            "scores": {k: 0 for k in _KEYS} | {"misunderstanding_risk": 100},
            "avg": 0.0,
            "passed": False,
            "per_provider": results,
            "feedback": "All quality judges failed; gather better domain sources.",
        }
    avg = mean_scores(ok)
    passed = passes(avg)
    feedback = ""
    if not passed:
        s = domain_settings
        thresholds = {
            "credibility": (avg.credibility, s.QUALITY_CREDIBILITY_MIN, True),
            "completeness": (avg.completeness, s.QUALITY_COMPLETENESS_MIN, True),
            "depth": (avg.depth, s.QUALITY_DEPTH_MIN, True),
            "terminology": (avg.terminology, s.QUALITY_TERMINOLOGY_MIN, True),
            "coherence": (avg.coherence, s.QUALITY_COHERENCE_MIN, True),
            "misunderstanding_risk": (avg.misunderstanding_risk, s.QUALITY_MISUNDERSTANDING_MAX, False),
        }
        weak = [
            k for k, (v, thr, ge) in thresholds.items()
            if (v < thr if ge else v > thr)
        ]
        feedback = (
            "Quality gate failed on: " + ", ".join(weak)
            + ". Prefer deeper sources with clear state→action→final-state."
        )
    return {
        "scores": avg.model_dump(),
        "avg": avg.avg(),
        "passed": passed,
        "per_provider": results,
        "feedback": feedback,
    }


async def run_quality_team(text: str, *, fallback_provider: str | None = None) -> dict:
    providers = [p.strip() for p in domain_settings.QUALITY_PROVIDERS.split(",") if p.strip()]
    if not providers:
        providers = [fallback_provider or llm_settings.CHAT_PROVIDER or "mock"]
    try:
        raw = await asyncio.wait_for(
            asyncio.gather(*[_judge(p, text) for p in providers], return_exceptions=True),
            timeout=domain_settings.QUALITY_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        raw = []
    results = [r if isinstance(r, dict) else {"error": str(r)} for r in raw]
    return aggregate_quality(results)


async def assess(text: str, *, provider: str | None = None) -> dict:
    """MCP / callers alias."""
    return await run_quality_team(text, fallback_provider=provider)
