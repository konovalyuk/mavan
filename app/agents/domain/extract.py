from app.agents.domain.loop import llm_json

EXTRACT_PROMPT = """Extract State-Action-Outcome transitions for decision intelligence training.
Focus on FINAL or stable system outcomes after an action — not intermediate steps.
Return JSON list:
[{{"state": "...", "action": "...", "outcome_states": [{{"state": "...", "prob_hint": 0.0-1.0}}]}}]

Text:
{text}
"""


def _outcomes(item: dict) -> list[dict]:
    raw = item.get("outcome_states") or []
    return [{"state": x["state"], "prob_hint": float(x.get("prob_hint", 0.5))}
            for x in raw if x.get("state")]


async def extract_transitions(text: str, quality_avg: float, *, provider: str | None = None) -> list[dict]:
    raw = await llm_json(EXTRACT_PROMPT.format(text=text[:12000]), provider=provider)
    if isinstance(raw, dict):
        raw = raw.get("transitions", raw.get("samples", []))
    if not isinstance(raw, list):
        return []

    out = []
    for item in raw:
        outcomes = _outcomes(item)
        if item.get("state") and item.get("action") and outcomes:
            out.append({"state": item["state"], "action": item["action"],
                        "outcome_states": outcomes, "quality_avg": quality_avg})
    return out
