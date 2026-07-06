#!/usr/bin/env python3
"""Agent eval: tool called + non-empty answer."""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import rag_settings
from app.agents.loop import run_tool_loop


async def main():
    gt_path = Path(rag_settings.ground_truth_path)
    rows = [json.loads(l) for l in gt_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    tool_called = 0
    answered = 0
    log = []

    for row in rows[:10]:
        answer, tool_log = await run_tool_loop(row["question"])
        called = len(tool_log) > 0
        ok = bool(answer.strip())
        tool_called += int(called)
        answered += int(ok)
        log.append({"question": row["question"], "tool_called": called, "answered": ok, "tool_log": tool_log})
        print(f"tools={called}  answered={ok}  q={row['question'][:50]!r}")

    n = len(log) or 1
    summary = {
        "tool_call_rate": tool_called / n,
        "answer_rate": answered / n,
        "n": n,
    }
    print(f"\nTool call rate: {summary['tool_call_rate']:.3f}")
    print(f"Answer rate:    {summary['answer_rate']:.3f}")

    out = Path(rag_settings.EVAL_DIR) / "agent_eval.json"
    out.write_text(json.dumps({"summary": summary, "log": log}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
