from pathlib import Path

import pytest

from chunklab.loaders.text import TextLoader
from chunklab.models import Document

EXAMPLES = Path(__file__).parent.parent / "examples"
TEST_DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def handbook() -> Document:
    return TextLoader().load(TEST_DATA / "employee_handbook.md")


@pytest.fixture()
def fake_embedder():
    from chunklab.embeddings.fake import FakeEmbedder

    return FakeEmbedder()


class CountingEmbedder:
    """FakeEmbedder that records what it was actually asked to compute.

    Used by the cache tests: the question is never "is it fast" but "did it call
    the model at all", which only a counting double can answer.
    """

    def __init__(self, revision: str | None = "rev-a", cache_signature: str = "") -> None:
        from chunklab.embeddings.fake import FakeEmbedder

        self._inner = FakeEmbedder()
        self.model_name = "counting-embedder"
        self.max_seq_tokens = None
        self.revision = revision
        self.cache_signature = cache_signature
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]):
        self.calls.append(list(texts))
        return self._inner.embed(texts)

    def embed_queries(self, texts: list[str]):
        self.calls.append(list(texts))
        return self._inner.embed(texts)
