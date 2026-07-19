from __future__ import annotations

from app.agents.decision.context import DecisionCtx
from app.agents.decision.decision_tools import (
    DECISION_TOOLS,
    FORECAST_TOOLS,
    GET_DECISION_STATUS_TOOL,
    execute_decision,
    execute_decision_status,
    execute_forecast,
)
from app.agents.runtime import (
    AgentLoopResult,
    Agent,
    agent_as_tool_schema,
    make_agent_tool,
    resolve_provider,
    run_agent,
)
from config import domain_settings

FORECAST_AGENT = "ForecastAgent"
DECISION_AGENT = "DecisionAgent"
COORDINATOR = "DecisionCoordinator"


def _forecast_agent(provider: str) -> Agent:
    return Agent(
        name=FORECAST_AGENT,
        goal="Produce a calibrated Action→State forecast using domain prior + LLM ensemble.",
        instruction=(
            "1) predict_domain for priors from the personalized model.\n"
            "2) forecast_ensemble for multi-LLM estimates.\n"
            "3) merge_forecasts to fuse prior and llm_avg.\n"
            "Stop when forecast is ready."
        ),
        tools=FORECAST_TOOLS,
        provider=provider,
        max_rounds=domain_settings.SPECIALIST_AGENT_MAX_ROUNDS,
    )


def _decision_agent(provider: str) -> Agent:
    return Agent(
        name=DECISION_AGENT,
        goal="Recommend the best action given the forecast and risk_aversion.",
        instruction=(
            "Call get_forecast if needed, then analyze_forecast. "
            "Stop when a recommended_action is available."
        ),
        tools=DECISION_TOOLS,
        provider=provider,
        max_rounds=domain_settings.SPECIALIST_AGENT_MAX_ROUNDS,
    )


def _coordinator_agent(provider: str) -> Agent:
    return Agent(
        name=COORDINATOR,
        goal="Return a decision recommendation with an underlying forecast.",
        instruction=(
            "Call ForecastAgent first, then DecisionAgent. "
            "Use get_decision_status to observe progress. "
            "Stop when recommendation exists."
        ),
        tools=[
            agent_as_tool_schema(FORECAST_AGENT, "Domain prior + LLM ensemble forecast specialist"),
            agent_as_tool_schema(DECISION_AGENT, "Decision analysis specialist"),
            GET_DECISION_STATUS_TOOL,
        ],
        provider=provider,
        max_rounds=domain_settings.DECISION_AGENT_MAX_ROUNDS,
    )


async def run_decision_recommend(ctx: DecisionCtx) -> AgentLoopResult:
    """LLM-manager coordinator + Forecast/Decision specialists (Stage 2)."""
    provider = resolve_provider(ctx.provider, domain_settings.COORDINATOR_PROVIDER)
    forecast = make_agent_tool(
        _forecast_agent(provider),
        lambda name, args: execute_forecast(ctx, name, args),
    )
    decision = make_agent_tool(
        _decision_agent(provider),
        lambda name, args: execute_decision(ctx, name, args),
    )

    async def execute(name: str, args: dict) -> tuple[str, dict]:
        if name == FORECAST_AGENT:
            text, meta = await forecast(name, args)
            ctx.specialist_logs.append(meta)
            return text, meta
        if name == DECISION_AGENT:
            text, meta = await decision(name, args)
            ctx.specialist_logs.append(meta)
            return text, meta
        if name == "get_decision_status":
            return await execute_decision_status(ctx, name, args)
        raise ValueError(f"Unknown coordinator tool: {name}")

    user = (
        f"Recommend an action for domain_id={ctx.domain_id}.\n"
        f"context_state={ctx.context_state!r}\n"
        f"candidate_actions={ctx.candidate_actions!r}\n"
        f"risk_aversion={ctx.risk_aversion}"
    )
    return await run_agent(_coordinator_agent(provider), user, execute)
