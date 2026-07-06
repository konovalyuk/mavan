import math


def domain_confidence(prior: dict[str, float]) -> float:
    if not prior:
        return 0.0
    total = sum(prior.values())
    if total <= 0:
        return 0.0
    probs = [p / total for p in prior.values()]
    peak = max(probs)
    n = len(probs)
    if n <= 1:
        return peak
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    return peak * (1.0 - entropy / math.log(n))


def normalize(dist: dict[str, float]) -> dict[str, float]:
    total = sum(dist.values())
    return {k: v / total for k, v in dist.items()} if total > 0 else dist


def average_llm(results: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    acc: dict[str, dict[str, list[float]]] = {}
    for fc in results:
        for action, states in fc.items():
            acc.setdefault(action, {})
            for st, p in states.items():
                acc[action].setdefault(st, []).append(float(p))
    return {a: normalize({st: sum(v) / len(v) for st, v in sts.items()}) for a, sts in acc.items()}


def merge_forecasts(
    prior_by_action: dict[str, dict[str, float]],
    llm_by_action: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    out = {}
    for action in set(prior_by_action) | set(llm_by_action):
        prior, llm = prior_by_action.get(action, {}), llm_by_action.get(action, {})
        alpha = domain_confidence(prior)
        merged = {
            st: alpha * prior.get(st, 0) + (1 - alpha) * llm.get(st, 0)
            for st in set(prior) | set(llm)
        }
        out[action] = normalize(merged) if merged else {"unknown": 1.0}
    return out
