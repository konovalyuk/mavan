#!/usr/bin/env python3
"""Compare no-rag | rag | agent with rate-limit friendly delays."""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import rag_settings
from app.rag.pipeline import get_default_pipeline
from app.agents.loop import run_tool_loop
from app.llm.capabilities import Capability, get_capability
from app.llm.chat.chat_providers import prepare_chat_request
from app.llm.chat.schemas import ChatCompletionRequest, ChatMessage

JUDGE = """Score 1 if ANSWER is supported by REFERENCE text, else 0.
Reply JSON: {{"score": 0 or 1}}

QUESTION: {q}
REFERENCE: {ref}
ANSWER: {a}
"""
SLEEP_SEC = float(__import__("os").getenv("EVAL_LLM_SLEEP_SEC", "5"))


async def llm_complete(chat, req):
    for attempt in range(3):
        try:
            return await chat.complete(req)
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                await asyncio.sleep(SLEEP_SEC * (attempt + 1))
                continue
            raise
    return None


async def judge(chat, question: str, reference: str, answer: str) -> int:
    req = prepare_chat_request(ChatCompletionRequest(
        messages=[ChatMessage(role="user", content=JUDGE.format(q=question, ref=reference[:1500], a=answer))],
        temperature=0.0,
    ))
    resp = await llm_complete(chat, req)
    try:
        return int(json.loads(resp.text or "{}").get("score", 0))
    except (json.JSONDecodeError, ValueError, AttributeError):
        return 0


async def main():
    rows = [json.loads(l) for l in Path(rag_settings.ground_truth_path).read_text().splitlines() if l.strip()]
    chat = get_capability(Capability.CHAT)(None)
    pipeline = get_default_pipeline()
    results = {"no_rag": [], "rag": [], "agent": []}

    for row in rows[:10]:
        q = row["question"]
        ref = next((c.text for c in (await pipeline.retrieve_chunks(q, top_k=1))), "")

        plain = await llm_complete(chat, prepare_chat_request(
            ChatCompletionRequest(messages=[ChatMessage(role="user", content=q)], temperature=0.2),
        ))
        await asyncio.sleep(SLEEP_SEC)
        rag = await pipeline.answer(q)
        await asyncio.sleep(SLEEP_SEC)
        agent_ans, _ = await run_tool_loop(q)
        await asyncio.sleep(SLEEP_SEC)

        results["no_rag"].append(await judge(chat, q, ref, plain.text or ""))
        results["rag"].append(await judge(chat, q, ref, rag.text or ""))
        results["agent"].append(await judge(chat, q, ref, agent_ans))

    summary = {k: round(sum(v) / len(v), 3) if v else 0 for k, v in results.items()}
    print(summary)
    out = Path(rag_settings.EVAL_DIR) / "llm_approaches_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
