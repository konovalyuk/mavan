#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import rag_settings
from app.rag.stores.file_store import load_and_chunk, save_chunk_index


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG chunk index from text files")
    parser.add_argument("--data-dir", type=Path, default=Path(rag_settings.DATA_DIR))
    parser.add_argument("--out", type=Path, default=Path(rag_settings.INDEX_PATH))
    args = parser.parse_args()

    data_dir = resolve_project_path(args.data_dir)
    out_path = resolve_project_path(args.out)

    if not data_dir.is_dir():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    chunks = load_and_chunk(data_dir)
    if not chunks:
        print(f"No chunks found in {data_dir}")
        print(f"Hint: add .txt or .md files under {data_dir}")
        sys.exit(1)

    save_chunk_index(out_path, chunks)
    print(f"Indexed {len(chunks)} chunks -> {out_path}")


if __name__ == "__main__":
    main()
