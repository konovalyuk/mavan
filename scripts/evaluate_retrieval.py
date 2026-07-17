#!/usr/bin/env python3
"""Compare retrievers: file | vector | hybrid | hybrid+rerank"""
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import rag_settings
from app.eval.retrieval_metrics import hit_rate, mrr, relevance_flags
from app.rag.pipeline import RagPipeline, build_retriever


def load_gt() -> list[dict]:
    path = Path(rag_settings.ground_truth_path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def eval_retriever(pipeline: RagPipeline, gt: list[dict], *, top_k: int = 5) -> dict:
    hit_rates, mrrs = [], []
    for row in gt:
        results = await pipeline.retrieve_chunks(row["question"], top_k=top_k)
        rel = relevance_flags(results, row, top_k=top_k)
        hit_rates.append(hit_rate(rel))
        mrrs.append(mrr(rel))
    n = len(gt) or 1
    return {"hit_rate": sum(hit_rates) / n, "mrr": sum(mrrs) / n, "n": n}


async def run_config(*, retriever: str, rerank: bool) -> dict:
    os.environ["RAG_RETRIEVER"] = retriever
    os.environ["RAG_RERANK_ENABLED"] = "true" if rerank else "false"
    import importlib
    import config
    importlib.reload(config)
    from app.rag.pipeline import build_retriever as br

    pipeline = RagPipeline(retriever=br())
    label = f"{retriever}+rerank" if rerank else retriever
    metrics = await eval_retriever(pipeline, gt)
    return {"label": label, **metrics}


async def main():
    global gt
    gt = load_gt()
    if not gt:
        print(f"No ground truth at {rag_settings.ground_truth_path}")
        print("Run: python scripts/generate_ground_truth.py")
        return

    configs = [("file", False), ("vector", False), ("hybrid", False), ("hybrid", True)]
    results = []
    for retriever, rerank in configs:
        r = await run_config(retriever=retriever, rerank=rerank)
        results.append(r)
        print(f"{r['label']:16}  hit_rate={r['hit_rate']:.3f}  mrr={r['mrr']:.3f}  n={r['n']}")

    out_path = Path(rag_settings.EVAL_DIR) / "retrieval_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
