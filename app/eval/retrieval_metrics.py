def hit_rate(relevances: list[int]) -> float:
    # relevances = [0,0,1,0,0] для top-5 результатов
    # Hit Rate = 1 если хотя бы один relevant в top-k, иначе 0
    return 1.0 if any(relevances) else 0.0


def mrr(relevances: list[int]) -> float:
    # MRR = 1/rank первого relevant документа
    # [0,0,1,...] → MRR = 1/3 = 0.333
    # [0,0,0,0,0] → MRR = 0
    for i, rel in enumerate(relevances, start=1):
        if rel:
            return 1.0 / i
    return 0.0


def compute_relevance(*, expected_source: str, results: list, top_k: int = 5) -> list[int]:
    out = []
    for doc in results[:top_k]:
        out.append(1 if doc.source == expected_source else 0)
    return out


def evaluate_search(fn, ground_truth: list[dict]) -> dict:
    hrs, mrrs = [], []
    for row in ground_truth:
        results = fn(row["question"])  # sync wrapper or precomputed
        rel = compute_relevance(expected_source=row["source"], results=results)
        hrs.append(hit_rate(rel))
        mrrs.append(mrr(rel))
    return {"hit_rate": sum(hrs) / len(hrs), "mrr": sum(mrrs) / len(mrrs)}
