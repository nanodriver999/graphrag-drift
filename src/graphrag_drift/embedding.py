from __future__ import annotations

import hashlib
import math
import re

DEFAULT_EMBEDDING_DIMENSIONS = 64


def hash_embedding(text: str, dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS) -> list[float]:
    """Deterministic lexical embedding for integration tests.

    Production deployments should replace this with a semantic embedding model.
    """
    vector = [0.0] * dimensions
    tokens = re.findall(r"[A-Za-z0-9가-힣_]+", text.lower())
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector
