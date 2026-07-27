"""Embedding backend registry."""

from chunklab.embeddings.base import Embedder


def make_embedder(backend: str, model: str, cache: bool = True) -> Embedder:
    if backend == "local":
        from chunklab.embeddings.cache import CachedEmbedder, caching_enabled
        from chunklab.embeddings.local import LocalEmbedder

        embedder = LocalEmbedder(model)
        # `fake` is already cheap and deterministic; caching it would only put
        # test vectors in the user's cache.
        if cache and caching_enabled():
            return CachedEmbedder(embedder)
        return embedder
    if backend == "fake":  # tests / dry runs
        from chunklab.embeddings.fake import FakeEmbedder

        return FakeEmbedder()
    if backend == "openai":
        raise NotImplementedError("openai backend is a fast-follow; use backend: local")
    raise ValueError(f"unknown embedding backend '{backend}'")
