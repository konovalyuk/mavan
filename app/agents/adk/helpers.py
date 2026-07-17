from google.adk.events.event import Event
from google.adk.utils.content_utils import extract_text_from_content


def extract_final_text(events: list[Event]) -> str:
    for event in reversed(events):
        if event.author == "user":
            continue
        text = extract_text_from_content(event.content)
        if text.strip():
            return text
    return ""


def extract_grounding_sources(events: list[Event]) -> list[dict]:
    seen: set[str] = set()
    sources: list[dict] = []

    for event in events:
        meta = event.grounding_metadata
        if not meta or not meta.grounding_chunks:
            continue
        for chunk in meta.grounding_chunks:
            web = chunk.web
            if not web or not web.uri or web.uri in seen:
                continue
            seen.add(web.uri)
            sources.append({"title": web.title or "", "url": web.uri})

    return sources
