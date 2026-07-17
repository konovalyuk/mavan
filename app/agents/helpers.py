from app.rag.types import RetrievedChunk


def chunks_to_sources(chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "text": c.text,
            "score": c.score,
            "source": c.source,
            "chunk_id": c.chunk_id,
        }
        for c in chunks
    ]


def sources_from_tool_log(tool_log: list[dict]) -> list[dict]:
    seen: set[str] = set()
    sources: list[dict] = []
    for entry in tool_log:
        for item in entry.get("meta", {}).get("sources", []):
            if isinstance(item, dict):
                key = item.get("chunk_id") or item.get("source") or item.get("url") or str(item)
                if key in seen:
                    continue
                seen.add(key)
                sources.append(item)
            elif isinstance(item, str):
                if item in seen:
                    continue
                seen.add(item)
                sources.append({"source": item})
    return sources
