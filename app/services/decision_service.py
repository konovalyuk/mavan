from fastapi import HTTPException

from app.agents.decision.context import DecisionCtx
from app.agents.decision.decision_agents import run_decision_recommend
from app.core.guardrails import validate_forecast
from app.domain.schemas import DecisionRequest


async def recommend(body: DecisionRequest, provider: str | None = None) -> dict:
    validate_forecast(body.context_state, body.candidate_actions)
    ctx = DecisionCtx(domain_id=body.domain_id, context_state=body.context_state, candidate_actions=body.candidate_actions, risk_aversion=body.risk_aversion, provider=provider)
    result = await run_decision_recommend(ctx)
    if not ctx.recommendation:
        raise HTTPException(status_code=400, detail=result.answer or "DecisionCoordinator did not produce a recommendation")
    out = dict(ctx.recommendation)
    out["coordinator_answer"] = result.answer
    out["coordinator_rounds"] = result.rounds
    out["tool_log"] = result.tool_log
    return out
