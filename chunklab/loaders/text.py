"""Loader for .txt and .md files."""

from pathlib import Path

from chunklab.loaders.base import make_doc_id
from chunklab.loaders.markdown_elements import extract_markdown_elements
from chunklab.models import Document


class TextLoader:
    def load(self, path: Path) -> Document:
        text = path.read_text(encoding="utf-8")
        elements = extract_markdown_elements(text) if path.suffix.lower() == ".md" else []
        return Document(
            id=make_doc_id(path),
            source_path=str(path),
            text=text,
            elements=elements,
            metadata={"format": path.suffix.lstrip(".").lower() or "txt"},
        )
