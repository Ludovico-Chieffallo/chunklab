"""On-disk embedding cache (roadmap 6.2).

The cache is only worth having if it is invisible: same vectors, fewer model
calls. These tests pin that, and pin the one way a cache can corrupt an
evaluation - serving vectors produced by different model weights.
"""

import numpy as np
import pytest

from chunklab.embeddings.cache import CachedEmbedder, EmbeddingCache
from chunklab.embeddings.fake import FakeEmbedder


class CountingEmbedder(FakeEmbedder):
    """FakeEmbedder that records what it was actually asked to compute."""

    def __init__(self, revision: str | None = "rev-a") -> None:
        super().__init__()
        self.model_name = "counting-embedder"
        self.revision = revision
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return super().embed(texts)


@pytest.fixture
def cache(tmp_path):
    store = EmbeddingCache(tmp_path / "vectors.sqlite3")
    yield store
    store.close()


def test_cached_vectors_are_identical_to_uncached(cache):
    texts = ["retention is thirty days", "invoices are payable in 30 days", "unrelated text"]
    plain = FakeEmbedder().embed(texts)

    embedder = CachedEmbedder(CountingEmbedder(), cache)
    first = embedder.embed(texts)
    second = embedder.embed(texts)

    np.testing.assert_array_equal(first, plain)
    np.testing.assert_array_equal(second, plain)


def test_second_pass_calls_no_model(cache):
    inner = CountingEmbedder()
    embedder = CachedEmbedder(inner, cache)
    texts = ["alpha beta", "gamma delta"]

    embedder.embed(texts)
    embedder.embed(texts)

    assert len(inner.calls) == 1, "the second pass recomputed vectors"
    assert embedder.hits == 2 and embedder.misses == 2


def test_repeated_text_in_one_batch_is_embedded_once(cache):
    inner = CountingEmbedder()
    embedder = CachedEmbedder(inner, cache)

    result = embedder.embed(["same text", "other", "same text"])

    assert inner.calls == [["same text", "other"]]
    np.testing.assert_array_equal(result[0], result[2])


def test_partial_hit_only_computes_the_new_texts(cache):
    inner = CountingEmbedder()
    embedder = CachedEmbedder(inner, cache)

    embedder.embed(["one", "two"])
    embedder.embed(["two", "three"])

    assert inner.calls[1] == ["three"]


def test_cache_survives_a_new_process(tmp_path):
    path = tmp_path / "vectors.sqlite3"
    first_store = EmbeddingCache(path)
    CachedEmbedder(CountingEmbedder(), first_store).embed(["persisted text"])
    first_store.close()

    inner = CountingEmbedder()
    second_store = EmbeddingCache(path)
    CachedEmbedder(inner, second_store).embed(["persisted text"])
    second_store.close()

    assert inner.calls == [], "a fresh process ignored the cache on disk"


def test_a_new_model_revision_is_not_served_from_the_old_one(cache):
    """The failure mode that matters: same text, different weights."""
    old = CountingEmbedder(revision="rev-a")
    CachedEmbedder(old, cache).embed(["shared text"])

    new = CountingEmbedder(revision="rev-b")
    CachedEmbedder(new, cache).embed(["shared text"])

    assert new.calls == [["shared text"]], "vectors leaked across model revisions"


def test_unknown_revision_does_not_collide_with_a_known_one(cache):
    known = CountingEmbedder(revision="rev-a")
    CachedEmbedder(known, cache).embed(["shared text"])

    unknown = CountingEmbedder(revision=None)
    CachedEmbedder(unknown, cache).embed(["shared text"])

    assert unknown.calls == [["shared text"]]


def test_empty_input_is_handled(cache):
    embedder = CachedEmbedder(CountingEmbedder(), cache)
    assert embedder.embed([]).shape[0] == 0
