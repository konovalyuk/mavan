from app.llm.chat.chat_providers import get_chat_provider
from app.llm.embed.embed_providers import get_embed_provider
from app.llm.rerank.rerank_providers import get_rerank_provider
from enum import Enum


class Capability(str, Enum):
    CHAT = "chat"
    EMBED = "embed"
    RERANK = "rerank"
    OCR = "ocr"
    SPEECH = "speech"
    IMAGE = "image"
    VISION = "vision"
    TASKS = "tasks"


_CAPABILITY_RESOLVERS = {
    Capability.CHAT: get_chat_provider,
    Capability.EMBED: get_embed_provider,
    Capability.RERANK: get_rerank_provider,
    # Capability.OCR: get_ocr_provider,
}


def get_capability(capability: Capability):
    resolver = _CAPABILITY_RESOLVERS.get(capability)
    if resolver is None:
        raise NotImplementedError(f"Capability {capability.value!r} is not implemented yet")
    return resolver
