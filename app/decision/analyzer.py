def analyze(forecast: dict[str, dict[str, float]], risk_aversion: float = 0.5) -> dict:
    """Score actions: higher prob on positive-sounding states wins; penalize risk keywords."""
    risk_words = {"fail", "loss", "crisis", "spike", "deficit", "unknown", "risk", "decline"}
    scores = {}
    for action, states in forecast.items():
        score = 0.0
        for state, prob in states.items():
            lower = state.lower()
            risk = any(w in lower for w in risk_words)
            utility = (1.0 - risk_aversion) if not risk else -risk_aversion
            score += prob * utility
        scores[action] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best = ranked[0][0] if ranked else None
    return {
        "recommended_action": best,
        "scores": scores,
        "rationale": f"Highest expected utility ({ranked[0][1]:.3f}) among {len(ranked)} actions." if ranked else "",
        "alternatives": [{"action": a, "score": s} for a, s in ranked[1:4]],
        "forecast": forecast,
    }
