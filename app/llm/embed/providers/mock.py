import hashlib
import math
import re


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class MockEmbedAdapter:
    """Deterministic bag-of-words vectors for local testing without API."""

    def __init__(self, *, dim: int = 256):
        self._dim = dim

    def _token_index(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest, 16) % self._dim

    def _vectorize(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in _tokenize(text):
            vec[self._token_index(token)] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        _ = model
        return [self._vectorize(t) for t in texts]
