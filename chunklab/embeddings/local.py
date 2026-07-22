"""Local embedder via sentence-transformers (default: BAAI/bge-small-en-v1.5)."""

import numpy as np

from chunklab.embeddings.base import normalize


class LocalEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.max_seq_tokens: int | None = getattr(self._model, "max_seq_length", None)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        )
        return normalize(np.asarray(vectors))
