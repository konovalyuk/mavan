import json
from dataclasses import asdict, dataclass
from pathlib import Path

from app.llm.embed.embed_providers import EmbedConfig

MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True)
class IndexManifest:
    embed_provider: str
    embed_model: str
    vector_dim: int
    chunk_count: int

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / MANIFEST_FILE).write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_dir: Path) -> "IndexManifest":
        path = index_dir / MANIFEST_FILE
        if not path.is_file():
            raise FileNotFoundError(
                f"Vector index manifest not found: {path}. "
                f"Rebuild index: python scripts/index_rag_vectors.py --from-json"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


class IndexStaleError(RuntimeError):
    pass


def assert_index_compatible(manifest: IndexManifest, current: EmbedConfig, *, vector_dim: int) -> None:
    if manifest.embed_provider != current.provider:
        raise IndexStaleError(
            f"Vector index was built with embed provider {manifest.embed_provider!r}, "
            f"but current config is {current.provider!r}. "
            f"Run: python scripts/index_rag_vectors.py --from-json"
        )

    if manifest.embed_model != current.model:
        raise IndexStaleError(
            f"Vector index was built with embed model {manifest.embed_model!r}, "
            f"but current config is {current.model!r}. "
            f"Run: python scripts/index_rag_vectors.py --from-json"
        )

    if manifest.vector_dim != vector_dim:
        raise IndexStaleError(
            f"Vector index dimension {manifest.vector_dim} "
            f"does not match current vectors ({vector_dim}). "
            f"Run: python scripts/index_rag_vectors.py --from-json"
        )
