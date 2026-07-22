"""PDF loader via pymupdf4llm (PDF -> markdown -> elements)."""

from pathlib import Path

from chunklab.loaders.base import make_doc_id
from chunklab.loaders.markdown_elements import extract_markdown_elements
from chunklab.models import Document


class PdfLoader:
    def load(self, path: Path) -> Document:
        import pymupdf4llm

        text = pymupdf4llm.to_markdown(str(path), show_progress=False)
        return Document(
            id=make_doc_id(path),
            source_path=str(path),
            text=text,
            elements=extract_markdown_elements(text),
            metadata={"format": "pdf"},
        )
