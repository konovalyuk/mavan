from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: float
    source: str | None = None  # "file:notes/intro.md"
    start_offset: int | None = None  # символ в исходном файле
    chunk_id: str | None = None  # стабильный ключ для RRF

    @property
    def rrf_key(self) -> tuple[str, int]:
        return (self.source or "unknown", self.start_offset or 0)


@dataclass(frozen=True)
class RagAnswer:
    text: str
    sources: list[RetrievedChunk]
