"""Embedder protocol."""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    model_name: str
    max_seq_tokens: int | None
    revision: str | None  # model revision/commit hash when resolvable, for provenance

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed passages (chunks, sentences). Unit-normalized (n, d) float32."""
        ...

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        """Embed search queries.

        Separate from `embed` because asymmetric models (E5, BGE) prefix the two
        sides differently; for symmetric models this is the same computation.
        """
        ...


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)
