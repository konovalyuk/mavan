from pathlib import Path
import numpy as np

from app.llm.capabilities import Capability, get_capability
from app.llm.embed.embed_providers import resolve_embed_config
from app.rag.index_manifest import IndexManifest
from app.rag.stores.file_store import load_and_chunk, load_chunk_index
from app.rag.stores.memory import VectorStore


async def build_vector_index(*, data_dir: Path | None = None, chunks_json: Path | None = None,
                             embed_provider_name: str | None = None) -> tuple[VectorStore, IndexManifest]:
    if chunks_json and chunks_json.is_file():
        chunks = load_chunk_index(chunks_json)
    elif data_dir is not None:
        chunks = load_and_chunk(data_dir)
    else:
        raise ValueError("Provide data_dir or chunks_json")

    if not chunks:
        raise ValueError("No chunks to index")

    cfg = resolve_embed_config(provider=embed_provider_name)
    provider = get_capability(Capability.EMBED)(cfg.provider)

    vectors = await provider.embed([c.text for c in chunks], model=cfg.model)
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.shape[0] != len(chunks):
        raise ValueError("Embed provider returned wrong number of vectors")

    manifest = IndexManifest(
        embed_provider=cfg.provider,
        embed_model=cfg.model,
        vector_dim=int(matrix.shape[1]),
        chunk_count=len(chunks),
    )
    return VectorStore(chunks=chunks, vectors=matrix), manifest
