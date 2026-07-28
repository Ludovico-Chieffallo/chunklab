"""How the embedder is assembled: local backend, caching, prefixes.

Every run goes through `make_embedder`, but nothing tested the `local` branch,
because doing so meant downloading a model. A stand-in for
`SentenceTransformer` covers the wiring — which layer wraps which, and what the
switches actually switch — without the download.
"""

import numpy as np
import pytest

from chunklab.embeddings.cache import CachedEmbedder
from chunklab.embeddings.local import LocalEmbedder
from chunklab.embeddings.registry import make_embedder


class FakeSentenceTransformer:
    def __init__(self, *_args, **_kwargs):
        self.max_seq_length = 512

    def encode(self, texts, **_kwargs):
        return np.ones((len(texts), 8), dtype="float32")


@pytest.fixture(autouse=True)
def no_model_download(monkeypatch, tmp_path):
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", FakeSentenceTransformer)
    # Never touch the user's real cache while testing.
    monkeypatch.setenv("CHUNKLAB_CACHE_DIR", str(tmp_path))


def test_local_backend_is_wrapped_in_the_cache_by_default():
    embedder = make_embedder("local", "BAAI/bge-small-en-v1.5")

    assert isinstance(embedder, CachedEmbedder)
    assert embedder.model_name == "BAAI/bge-small-en-v1.5"


def test_cache_can_be_turned_off():
    embedder = make_embedder("local", "BAAI/bge-small-en-v1.5", cache=False)

    assert isinstance(embedder, LocalEmbedder)


def test_environment_variable_overrides_the_config():
    """CHUNKLAB_NO_CACHE must win even when the config asks for caching."""
    import os

    os.environ["CHUNKLAB_NO_CACHE"] = "1"
    try:
        embedder = make_embedder("local", "BAAI/bge-small-en-v1.5", cache=True)
        assert isinstance(embedder, LocalEmbedder)
    finally:
        del os.environ["CHUNKLAB_NO_CACHE"]


def test_fake_backend_is_never_cached():
    """Caching the test double would only put test vectors in the user's cache."""
    embedder = make_embedder("fake", "irrelevant")

    assert not isinstance(embedder, CachedEmbedder)


def test_prefixes_are_applied_through_the_registry():
    embedder = make_embedder("local", "intfloat/multilingual-e5-small", cache=False)

    assert embedder.prefixes.query == "query: "
    assert embedder.prefixes.passage == "passage: "


def test_prefixes_can_be_disabled():
    embedder = make_embedder("local", "intfloat/multilingual-e5-small", cache=False, prefixes=False)

    assert not embedder.prefixes


def test_max_sequence_length_is_carried_through_the_cache():
    """chunk_health warns about oversized chunks using this; losing it silences the warning."""
    embedder = make_embedder("local", "BAAI/bge-small-en-v1.5")

    assert embedder.max_seq_tokens == 512


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown embedding backend"):
        make_embedder("telepathy", "irrelevant")


def test_openai_backend_says_it_is_not_implemented():
    with pytest.raises(NotImplementedError, match="fast-follow"):
        make_embedder("openai", "text-embedding-3-small")


def test_vectors_are_unit_normalized():
    """Cosine similarity is computed as a dot product, which is only correct for
    unit vectors."""
    embedder = make_embedder("local", "BAAI/bge-small-en-v1.5", cache=False)

    vectors = embedder.embed(["one", "two"])

    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6)
