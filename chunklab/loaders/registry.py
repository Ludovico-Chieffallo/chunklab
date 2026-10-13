"""Map file extensions to loaders, and load documents from paths."""

from pathlib import Path

from chunklab.errors import DocumentLoadError, InputError, InputNotFoundError
from chunklab.loaders.base import DocumentLoader
from chunklab.models import Document

# DocumentLoadError moved to chunklab.errors; importing it from here still works.
__all__ = ["DocumentLoadError", "SUPPORTED_EXTENSIONS", "get_loader", "load_documents"]

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def get_loader(extension: str) -> DocumentLoader:
    ext = extension.lower()
    if ext == ".pdf":
        from chunklab.loaders.pdf import PdfLoader

        return PdfLoader()
    if ext == ".docx":
        from chunklab.loaders.docx import DocxLoader

        return DocxLoader()
    if ext in {".txt", ".md"}:
        from chunklab.loaders.text import TextLoader

        return TextLoader()
    raise ValueError(
        f"unsupported file type '{extension}'; supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def load_documents(docs_path: str | Path) -> list[Document]:
    """Load a single file or every supported file in a directory (recursive)."""
    path = Path(docs_path)
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(
            p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    else:
        raise InputNotFoundError(f"docs path not found: {path}")
    if not files:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise InputError(f"no supported documents found under {path} (supported: {supported})")

    documents: list[Document] = []
    for file in files:
        try:
            documents.append(get_loader(file.suffix).load(file))
        except Exception as exc:  # a raw parser traceback does not name the file
            raise DocumentLoadError(f"could not read {file}: {type(exc).__name__}: {exc}") from exc
    return documents
