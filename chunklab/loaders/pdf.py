"""PDF loader via pymupdf4llm (PDF -> markdown -> elements).

pymupdf4llm and the PyMuPDF it runs on are licensed under the AGPL-3.0 (or a
commercial licence from Artifex), so they are not a dependency of chunklab
itself: they come with the `pdf` extra, and nothing else needs them.
"""

import importlib.util
from pathlib import Path

from chunklab.errors import InputError
from chunklab.loaders.base import make_doc_id
from chunklab.loaders.markdown_elements import extract_markdown_elements
from chunklab.models import Document


def _pdf_library_installed() -> bool:
    return importlib.util.find_spec("pymupdf4llm") is not None


def require_pdf_support(count: int = 1) -> None:
    """Raise one line naming the extra when `count` PDFs cannot be read."""
    if not _pdf_library_installed():
        raise InputError(
            f"{count} PDF file(s) to read, and reading PDFs needs the optional PDF extra: "
            "pip install 'chunklab[pdf]'. It is separate because its library, PyMuPDF, "
            "is licensed under the AGPL"
        )


class PdfLoader:
    def load(self, path: Path) -> Document:
        require_pdf_support()
        import pymupdf4llm

        text = pymupdf4llm.to_markdown(str(path), show_progress=False)
        return Document(
            id=make_doc_id(path),
            source_path=str(path),
            text=text,
            elements=extract_markdown_elements(text),
            metadata={"format": "pdf"},
        )
