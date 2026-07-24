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
