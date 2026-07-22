"""Map file extensions to loaders, and load documents from paths."""

from pathlib import Path

from chunklab.loaders.base import DocumentLoader
from chunklab.models import Document

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
        raise FileNotFoundError(f"docs path not found: {path}")
    if not files:
        raise ValueError(f"no supported documents found under {path}")
    return [get_loader(p.suffix).load(p) for p in files]
