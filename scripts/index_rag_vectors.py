#!/usr/bin/env python3
import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import rag_settings
from app.rag.index_store import upsert_chunks
from app.rag.pipeline import reset_pipeline
from app.rag.stores.file_store import chunk_text, load_chunk_index, load_text_files


def rp(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


async def main() -> None:
    p = argparse.ArgumentParser(description="Index corpus into RAG (vector + keyword)")
    p.add_argument("--data-dir", type=Path, default=Path(rag_settings.DATA_DIR))
    p.add_argument("--chunks-json", type=Path, default=None)
    p.add_argument("--from-json", action="store_true")
    p.add_argument("--strategy", choices=["paragraph", "sliding"], default="paragraph")
    p.add_argument("--namespace", default="file:", help="Replace chunks with this source prefix")
    args = p.parse_args()

    if args.from_json:
        chunks_path = rp(args.chunks_json) if args.chunks_json else rag_settings.chunks_path
        chunks = load_chunk_index(chunks_path)
    else:
        chunks = []
        for rel, text in load_text_files(rp(args.data_dir)):
            chunks.extend(chunk_text(text, source=f"file:{rel}", strategy=args.strategy))

    if not chunks:
        print("No chunks found")
        sys.exit(1)

    stats = await upsert_chunks(chunks, remove_prefixes=[args.namespace])
    reset_pipeline()
    print(f"Indexed {stats['chunks_added']} chunks, total={stats['total_chunks']}, namespace={args.namespace!r}")


if __name__ == "__main__":
    asyncio.run(main())
