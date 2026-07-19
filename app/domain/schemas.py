from typing import Literal

from pydantic import BaseModel, Field


class SourceCreate(BaseModel):
    url: str
    title: str = ""


class SourceOut(BaseModel):
    id: str
    domain_id: str
    url: str
    title: str = ""
    status: Literal["discovered", "approved", "rejected"] = "discovered"


class OutcomeStateHint(BaseModel):
    state: str
    prob_hint: float = 0.5


class TrainingSample(BaseModel):
    state: str
    action: str
    outcome_states: list[OutcomeStateHint] = Field(default_factory=list)
    quality_avg: float = 0.0


class QualityScores(BaseModel):
    credibility: float = 0.0
    completeness: float = 0.0
    depth: float = 0.0
    terminology: float = 0.0
    coherence: float = 0.0
    misunderstanding_risk: float = 0.0

    def avg(self) -> float:
        return (
                self.credibility + self.completeness + self.depth
                + self.terminology + self.coherence + (100 - self.misunderstanding_risk)
        ) / 6


class DecisionRequest(BaseModel):
    domain_id: str
    context_state: str
    candidate_actions: list[str]
    risk_aversion: float = 0.5
