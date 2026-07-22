"""DocumentLoader protocol."""

from pathlib import Path
from typing import Protocol, runtime_checkable

from chunklab.models import Document


@runtime_checkable
class DocumentLoader(Protocol):
    def load(self, path: Path) -> Document: ...


def make_doc_id(path: Path) -> str:
    return path.stem
