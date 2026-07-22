"""Embedder protocol."""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    model_name: str
    max_seq_tokens: int | None

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a unit-normalized (n, d) float32 array."""
        ...


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)
