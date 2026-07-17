import json
import httpx
import trafilatura

from app.core.guardrails import sanitize_text, validate_url
from config import domain_settings


async def google_search(query: str) -> tuple[str, dict]:
    if not domain_settings.GOOGLE_SEARCH_API_KEY:
        return json.dumps({"error": "GOOGLE_SEARCH_API_KEY not set", "results": []}), {}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            domain_settings.GOOGLE_SEARCH_URL,
            headers={"X-API-KEY": domain_settings.GOOGLE_SEARCH_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": 10},
        )
        r.raise_for_status()
        data = r.json()
    results = [{"url": o.get("link", ""), "title": o.get("title", ""), "snippet": o.get("snippet", "")}
               for o in data.get("organic", []) if o.get("link")]
    return json.dumps({"results": results}), {"count": len(results)}


def fetch_url_text(url: str) -> str | None:
    validate_url(url)
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None
    text = sanitize_text(trafilatura.extract(downloaded) or "")
    return text if text.strip() else None
