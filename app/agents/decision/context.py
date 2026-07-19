from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DecisionCtx:
    """Shared environment for Stage-2 specialist agents (not an agent)."""

    domain_id: str
    context_state: str
    candidate_actions: list[str]
    risk_aversion: float = 0.5
    provider: str | None = None
    prior: dict = field(default_factory=dict)
    llm_avg: dict = field(default_factory=dict)
    forecast: dict = field(default_factory=dict)
    recommendation: dict = field(default_factory=dict)
    specialist_logs: list[dict] = field(default_factory=list)
