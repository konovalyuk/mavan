from __future__ import annotations

import asyncio
import json

from app.agents.decision.context import DecisionCtx
from app.agents.runtime import tool_schema
from app.decision.analyzer import analyze
from app.decision.forecast import FORECAST_PROMPT, core_states, forecast_one
from app.decision.fuse import average_llm, merge_forecasts
from app.domain import store
from app.domain_model.inference import predict
from config import domain_settings

PREDICT_DOMAIN_TOOL = tool_schema(
    "predict_domain",
    "Get domain-model prior outcome distributions for each candidate action.",
    {"type": "object", "properties": {}},
)

FORECAST_ENSEMBLE_TOOL = tool_schema(
    "forecast_ensemble",
    "Query multiple LLM providers in parallel for Action→State forecasts, then average.",
    {"type": "object", "properties": {}},
)

MERGE_FORECASTS_TOOL = tool_schema(
    "merge_forecasts",
    "Confidence-weighted merge of domain prior and LLM ensemble into final forecast.",
    {"type": "object", "properties": {}},
)

GET_KNOWN_STATES_TOOL = tool_schema(
    "get_known_states",
    "List known outcome states from domain core_memory.",
    {"type": "object", "properties": {}},
)

GET_FORECAST_TOOL = tool_schema(
    "get_forecast",
    "Return the current merged forecast JSON.",
    {"type": "object", "properties": {}},
)

ANALYZE_FORECAST_TOOL = tool_schema(
    "analyze_forecast",
    "Score actions by expected utility under risk_aversion and pick a recommendation.",
    {"type": "object", "properties": {}},
)

GET_DECISION_STATUS_TOOL = tool_schema(
    "get_decision_status",
    "Observe whether prior/llm/forecast/recommendation are already computed.",
    {"type": "object", "properties": {}},
)

FORECAST_TOOLS = [
    PREDICT_DOMAIN_TOOL,
    FORECAST_ENSEMBLE_TOOL,
    MERGE_FORECASTS_TOOL,
    GET_KNOWN_STATES_TOOL,
    GET_FORECAST_TOOL,
]
DECISION_TOOLS = [GET_FORECAST_TOOL, ANALYZE_FORECAST_TOOL]


async def execute_forecast(ctx: DecisionCtx, name: str, args: dict) -> tuple[str, dict]:
    if name == "predict_domain":
        ctx.prior = {a: predict(ctx.domain_id, ctx.context_state, a) for a in ctx.candidate_actions}
        return json.dumps(ctx.prior), {"actions": len(ctx.prior)}

    if name == "get_known_states":
        states = await core_states(ctx.domain_id)
        return json.dumps(states), {"count": len(states)}

    if name == "forecast_ensemble":
        domain = await store.get_domain(ctx.domain_id) or {}
        known = await core_states(ctx.domain_id)
        prompt = FORECAST_PROMPT.format(
            name=domain.get("name", ctx.domain_id),
            description=domain.get("description", ""),
            known_states=json.dumps(known, ensure_ascii=False),
            context_state=ctx.context_state,
            actions=json.dumps(ctx.candidate_actions, ensure_ascii=False),
        )
        providers = [p.strip() for p in domain_settings.FORECAST_PROVIDERS.split(",") if p.strip()]
        if not providers:
            providers = [ctx.provider or "mock"]
        try:
            raw = await asyncio.wait_for(
                asyncio.gather(*[forecast_one(p, prompt) for p in providers], return_exceptions=True),
                timeout=domain_settings.FORECAST_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            raw = []
        llm_results = [r for r in raw if isinstance(r, dict) and r]
        ctx.llm_avg = (
            average_llm(llm_results)
            if llm_results
            else {a: {"unknown": 1.0} for a in ctx.candidate_actions}
        )
        return json.dumps(ctx.llm_avg), {"providers_ok": len(llm_results)}

    if name == "merge_forecasts":
        if not ctx.prior:
            ctx.prior = {a: predict(ctx.domain_id, ctx.context_state, a) for a in ctx.candidate_actions}
        if not ctx.llm_avg:
            return json.dumps({"error": "call forecast_ensemble first"}), {"ok": False}
        ctx.forecast = merge_forecasts(ctx.prior, ctx.llm_avg)
        return json.dumps(ctx.forecast), {"actions": len(ctx.forecast)}

    if name == "get_forecast":
        return json.dumps(ctx.forecast or {}), {"ready": bool(ctx.forecast)}

    raise ValueError(f"Unknown forecast tool: {name}")


async def execute_decision(ctx: DecisionCtx, name: str, args: dict) -> tuple[str, dict]:
    if name == "get_forecast":
        return json.dumps(ctx.forecast or {}), {"ready": bool(ctx.forecast)}

    if name == "analyze_forecast":
        if not ctx.forecast:
            return json.dumps({"error": "forecast missing — coordinator should run ForecastAgent first"}), {
                "ok": False,
            }
        ctx.recommendation = analyze(ctx.forecast, risk_aversion=ctx.risk_aversion)
        return json.dumps(ctx.recommendation), {
            "recommended_action": ctx.recommendation.get("recommended_action"),
        }

    raise ValueError(f"Unknown decision tool: {name}")


async def execute_decision_status(ctx: DecisionCtx, name: str, args: dict) -> tuple[str, dict]:
    if name != "get_decision_status":
        raise ValueError(name)
    payload = {
        "has_prior": bool(ctx.prior),
        "has_llm_avg": bool(ctx.llm_avg),
        "has_forecast": bool(ctx.forecast),
        "has_recommendation": bool(ctx.recommendation),
        "recommended_action": (ctx.recommendation or {}).get("recommended_action"),
    }
    return json.dumps(payload), payload
