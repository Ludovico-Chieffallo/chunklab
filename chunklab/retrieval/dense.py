"""Dense retrieval: numpy cosine top-k over an in-memory chunk index."""

import numpy as np

from chunklab.embeddings.base import Embedder
from chunklab.models import Chunk, RetrievedChunk


class DenseRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder) -> None:
        self.chunks = chunks
        self.embedder = embedder
        # vectors are unit-normalized, so cosine similarity == dot product
        self._matrix = embedder.embed([c.text for c in chunks])

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        q = self.embedder.embed([query])[0]
        scores = self._matrix @ q
        k = min(top_k, len(self.chunks))
        # stable sort so score ties break by chunk order, deterministically
        top = np.argsort(-scores, kind="stable")[:k]
        return [
            RetrievedChunk(chunk=self.chunks[i], score=float(scores[i]), rank=r + 1)
            for r, i in enumerate(top)
        ]
