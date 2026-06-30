from pathlib import Path

from app.llm.capabilities import Capability, get_capability
from app.llm.embed.embed_providers import resolve_embed_config
from app.rag.index_manifest import IndexManifest, assert_index_compatible
from app.rag.stores.memory import VectorStore
from app.rag.types import RetrievedChunk


class VectorRetriever:
    def __init__(self, index_dir: Path):
        self._store = VectorStore.load(index_dir)
        self._cfg = resolve_embed_config()
        self._manifest = IndexManifest.load(index_dir)
        assert_index_compatible(
            self._manifest,
            self._cfg,
            vector_dim=int(self._store.vectors.shape[1]),
        )

    async def retrieve(self, question: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        provider = get_capability(Capability.EMBED)(self._cfg.provider)
        query_vector = (await provider.embed([question], model=self._cfg.model))[0]

        if len(query_vector) != self._store.vectors.shape[1]:
            raise ValueError(
                f"Query vector dim {len(query_vector)} != index dim {self._store.vectors.shape[1]}. "
                f"Run: python scripts/index_rag_vectors.py --from-json"
            )

        return self._store.search(query_vector, top_k=top_k)