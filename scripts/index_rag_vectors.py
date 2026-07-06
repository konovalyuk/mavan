#!/usr/bin/env python3
import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import rag_settings
from app.rag.indexer import build_vector_index


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Build vector RAG index")
    parser.add_argument("--data-dir", type=Path, default=Path(rag_settings.DATA_DIR))
    parser.add_argument("--chunks-json", type=Path, default=Path(rag_settings.INDEX_PATH))
    parser.add_argument("--out", type=Path, default=Path(rag_settings.VECTOR_INDEX_PATH))
    parser.add_argument("--from-json", action="store_true", help="Use existing chunks.json")
    parser.add_argument("--embed-provider", type=str, default=None)
    parser.add_argument("--strategy", choices=["paragraph", "sliding"], default="paragraph")
    args = parser.parse_args()

    out_dir = resolve_project_path(args.out)

    if args.from_json:
        chunks_json = resolve_project_path(args.chunks_json)
        store, manifest = await build_vector_index(
            chunks_json=chunks_json,
            embed_provider_name=args.embed_provider,
        )
    else:
        data_dir = resolve_project_path(args.data_dir)
        store, manifest = await build_vector_index(
            data_dir=data_dir,
            embed_provider_name=args.embed_provider,
        )

    store.save(out_dir)
    manifest.save(out_dir)
    print(
        f"Vector index: {len(store.chunks)} chunks -> {out_dir} "
        f"({manifest.embed_provider}/{manifest.embed_model}, dim={manifest.vector_dim})"
    )


if __name__ == "__main__":
    asyncio.run(main())
