#!/usr/bin/env python3
"""LLM-as-judge: score RAG answers on ground truth questions."""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import rag_settings
from app.rag.pipeline import get_default_pipeline
from app.llm.capabilities import Capability, get_capability
from app.llm.chat.chat_providers import prepare_chat_request
from app.llm.chat.schemas import ChatCompletionRequest, ChatMessage

JUDGE_PROMPT = """\
You are an evaluator. Score if the ANSWER is supported by the CONTEXT (1=yes, 0=no).
Reply JSON only: {{"score": 0 or 1, "reason": "..."}}

QUESTION: {question}
CONTEXT: {context}
ANSWER: {answer}
"""


async def main():
    gt_path = Path(rag_settings.ground_truth_path)
    rows = [json.loads(l) for l in gt_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    pipeline = get_default_pipeline()
    judge = get_capability(Capability.CHAT)(None)
    scores = []

    for row in rows[:10]:  # cap for dev
        rag = await pipeline.answer(row["question"])
        context = "\n".join(c.text[:300] for c in rag.sources)
        request = prepare_chat_request(
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content=JUDGE_PROMPT.format(
                    question=row["question"],
                    context=context,
                    answer=rag.text,
                ))],
                temperature=0.0,
            )
        )
        resp = await judge.complete(request)
        try:
            score = json.loads(resp.text or "{}").get("score", 0)
        except json.JSONDecodeError:
            score = 0
        scores.append(score)
        print(f"q={row['question'][:60]!r}  score={score}")

    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"\nLLM-judge avg score: {avg:.3f}  (n={len(scores)})")

    out = Path(rag_settings.EVAL_DIR) / "rag_answer_eval.json"
    out.write_text(json.dumps({"avg_score": avg, "n": len(scores)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
