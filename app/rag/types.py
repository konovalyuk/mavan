from dataclasses import dataclass

@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: float
    source: str | None = None   # "static", "file:notes.txt", "chunk:3"

@dataclass(frozen=True)
class RagAnswer:
    text: str
    sources: list[RetrievedChunk]