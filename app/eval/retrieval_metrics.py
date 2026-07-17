def hit_rate(relevances: list[int]) -> float:
    return 1.0 if any(relevances) else 0.0


def mrr(relevances: list[int]) -> float:
    for i, rel in enumerate(relevances, start=1):
        if rel:
            return 1.0 / i
    return 0.0


def relevance_flags(results: list, row: dict, *, top_k: int = 5) -> list[int]:
    rel = []
    for doc in results[:top_k]:
        if row.get("chunk_id") and getattr(doc, "chunk_id", None) == row["chunk_id"]:
            rel.append(1)
        elif getattr(doc, "source", None) == row.get("source"):
            rel.append(1)
        else:
            rel.append(0)
    return rel
