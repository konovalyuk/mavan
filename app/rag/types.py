from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: float
    source: str | None = None
    start_offset: int | None = None
    chunk_id: str | None = None

    @property
    def rrf_key(self) -> tuple[str, int]:
        if self.chunk_id:
            return (self.chunk_id, 0)
        return (self.source or "unknown", self.start_offset or 0)
