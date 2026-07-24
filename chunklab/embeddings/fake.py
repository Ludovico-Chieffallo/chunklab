"""Deterministic bag-of-words embedder for tests and offline dry-runs.

Hashes each word into a fixed-size bucket vector, so texts sharing vocabulary
get similar vectors. No model download, fully deterministic.
"""

import hashlib
import re

import numpy as np

from chunklab.embeddings.base import normalize

_WORD_RE = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    model_name = "fake-hash-embedder"
    max_seq_tokens: int | None = None
    revision: str | None = None

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _bucket(self, word: str) -> int:
        return int.from_bytes(hashlib.md5(word.encode()).digest()[:4], "big") % self.dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in _WORD_RE.findall(text.lower()):
                vectors[i, self._bucket(word)] += 1.0
        return normalize(vectors)
