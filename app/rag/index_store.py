import logging
from pathlib import Path
import numpy as np

from config import rag_settings
from app.llm.capabilities import Capability, get_capability
from app.llm.embed.embed_providers import resolve_embed_config
from app.rag.index_manifest import IndexManifest
from app.rag.sources import matches_source_prefix
from app.rag.stores.file_store import load_chunk_index
from app.rag.stores.memory import VectorStore
from app.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)


def _drop_by_prefixes(chunks: list[RetrievedChunk], prefixes: list[str]) -> list[RetrievedChunk]:
    if not prefixes:
        return chunks
    return [c for c in chunks if not matches_source_prefix(c.source, prefixes)]


def _kept_vector_rows(store: VectorStore, remove_prefixes: list[str]) -> tuple[list[RetrievedChunk], np.ndarray]:
    rows = [i for i, c in enumerate(store.chunks) if not matches_source_prefix(c.source, remove_prefixes)]
    chunks = [store.chunks[i] for i in rows]
    dim = store.vectors.shape[1]
    matrix = store.vectors[rows] if rows else np.zeros((0, dim), dtype=np.float32)
    return chunks, matrix


async def upsert_chunks(new_chunks: list[RetrievedChunk], *, remove_prefixes: list[str]) -> dict:
    if not new_chunks and not remove_prefixes:
        return {"chunks_added": 0, "total_chunks": 0}

    vector_dir = Path(rag_settings.VECTOR_INDEX_PATH)
    vectors_path = vector_dir / "vectors.npy"
    manifest: IndexManifest | None = None

    if vectors_path.is_file():
        store = VectorStore.load(vector_dir)
        merged, merged_matrix = _kept_vector_rows(store, remove_prefixes)
        old_manifest = IndexManifest.load(vector_dir)
    else:
        chunks_path = rag_settings.chunks_path
        existing = load_chunk_index(chunks_path) if chunks_path.is_file() else []
        merged = _drop_by_prefixes(existing, remove_prefixes)
        merged_matrix = np.zeros((0, 0), dtype=np.float32)
        old_manifest = None

    if new_chunks:
        cfg = resolve_embed_config(provider=None)
        embed_provider = get_capability(Capability.EMBED)(cfg.provider)
        new_matrix = np.asarray(await embed_provider.embed([c.text for c in new_chunks], model=cfg.model), dtype=np.float32)
        merged = merged + new_chunks
        merged_matrix = new_matrix if merged_matrix.size == 0 else np.vstack([merged_matrix, new_matrix])
        manifest = IndexManifest(embed_provider=cfg.provider, embed_model=cfg.model, vector_dim=int(new_matrix.shape[1]), chunk_count=len(merged))
    elif old_manifest is not None:
        manifest = IndexManifest(embed_provider=old_manifest.embed_provider, embed_model=old_manifest.embed_model, vector_dim=int(merged_matrix.shape[1]), chunk_count=len(merged))

    if merged_matrix.shape[0] != len(merged):
        raise ValueError(f"chunks/vectors mismatch: {len(merged)} vs {merged_matrix.shape[0]}")

    VectorStore(chunks=merged, vectors=merged_matrix).save(vector_dir)
    if manifest:
        manifest.save(vector_dir)

    return {"chunks_added": len(new_chunks), "total_chunks": len(merged)}
