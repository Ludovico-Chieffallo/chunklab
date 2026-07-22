"""Embedding backend registry."""

from chunklab.embeddings.base import Embedder


def make_embedder(backend: str, model: str) -> Embedder:
    if backend == "local":
        from chunklab.embeddings.local import LocalEmbedder

        return LocalEmbedder(model)
    if backend == "fake":  # tests / dry runs
        from chunklab.embeddings.fake import FakeEmbedder

        return FakeEmbedder()
    if backend == "openai":
        raise NotImplementedError("openai backend is a fast-follow; use backend: local")
    raise ValueError(f"unknown embedding backend '{backend}'")
