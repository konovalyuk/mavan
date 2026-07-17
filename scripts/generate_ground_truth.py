#!/usr/bin/env python3
"""Generate Q&A pairs from indexed chunks → RAG_EVAL_DIR/ground_truth.jsonl"""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import rag_settings
from app.rag.stores.file_store import load_chunk_index
from app.llm.capabilities import Capability, get_capability
from app.llm.chat.chat_providers import prepare_chat_request
from app.llm.chat.schemas import ChatCompletionRequest, ChatMessage


def build_prompt(chunk_text: str) -> str:
    return (
        "Given this text chunk, write ONE factual question answerable ONLY from this text.\n"
        'Reply as JSON: {"question": "..."}\n\n'
        f"Chunk:\n{chunk_text[:1500]}"
    )


async def main():
    chunks = load_chunk_index(rag_settings.chunks_path)
    out_path = Path(rag_settings.ground_truth_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chat = get_capability(Capability.CHAT)(None)
    rows = []

    for chunk in chunks[:20]:
        request = prepare_chat_request(
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content=build_prompt(chunk.text))],
                temperature=0.2,
            )
        )
        response = await chat.complete(request)
        try:
            q = json.loads(response.text or "{}")["question"]
        except (json.JSONDecodeError, KeyError):
            continue
        rows.append({
            "question": q,
            "source": chunk.source,
            "chunk_id": chunk.chunk_id,
        })

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
