from app.core.guardrails import validate_forecast
from app.decision.analyzer import analyze
from app.decision.forecast import run_forecast
from app.domain.schemas import DecisionRequest, ForecastRequest


async def forecast(body: ForecastRequest, *, provider: str | None = None) -> dict:
    validate_forecast(body.context_state, body.candidate_actions)
    result = await run_forecast(
        body.domain_id, body.context_state, body.candidate_actions, provider=provider,
    )
    return {"forecast": result}


async def recommend(body: DecisionRequest, *, provider: str | None = None) -> dict:
    validate_forecast(body.context_state, body.candidate_actions)
    fc = await run_forecast(
        body.domain_id, body.context_state, body.candidate_actions, provider=provider,
    )
    return analyze(fc, risk_aversion=body.risk_aversion)
