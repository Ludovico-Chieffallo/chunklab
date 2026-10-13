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


@pytest.fixture()
def cli(monkeypatch):
    """Invoke the real `chunklab` CLI in-process and return the click Result.

    The module-level console is swapped for one wide enough not to wrap, so a
    test can assert on a whole line; it still writes to whatever stdout the
    runner installed, which is what the user would have seen.
    """
    from rich.console import Console
    from typer.testing import CliRunner

    import chunklab.cli as cli_module

    monkeypatch.setattr(cli_module, "console", Console(width=200))
    runner = CliRunner()

    def invoke(*args: str):
        return runner.invoke(cli_module.app, [str(arg) for arg in args])

    return invoke


@pytest.fixture()
def fake_config(tmp_path) -> Path:
    """A config that runs every strategy on the deterministic offline embedder."""
    path = tmp_path / "fake.yaml"
    path.write_text("embedding:\n  backend: fake\n", encoding="utf-8")
    return path


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
