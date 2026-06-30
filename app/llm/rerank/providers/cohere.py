import httpx

from app.rag.types import RetrievedChunk


class CohereRerankAdapter:
    def __init__(self, *, api_key: str):
        self._api_key = api_key

    async def rerank(
            self,
            query: str,
            chunks: list[RetrievedChunk],
            *,
            model: str,
            top_k: int,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.cohere.com/v1/rerank",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "query": query,
                    "documents": [c.text for c in chunks],
                    "top_n": top_k,
                },
            )
            response.raise_for_status()
            data = response.json()

        out: list[RetrievedChunk] = []
        for item in data["results"]:
            chunk = chunks[item["index"]]
            out.append(RetrievedChunk(
                text=chunk.text,
                score=float(item["relevance_score"]),
                source=chunk.source,
            ))
        return out
