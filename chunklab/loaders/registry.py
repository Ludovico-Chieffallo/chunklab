"""Map file extensions to loaders, and load documents from paths."""

from collections import Counter
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


def _not_meant_for_evaluation(path: Path, root: Path) -> bool:
    """Hidden files and folders, and the lock files office suites write beside
    an open document (Word's `~$name.docx`; LibreOffice's `.~lock.name#` is
    hidden already).

    A folder containing one used to fail the whole run with a parser error for a
    file the user cannot see and never chose.
    """
    relative = path.relative_to(root)
    return path.name.startswith("~$") or any(part.startswith(".") for part in relative.parts)


def describe_skipped(skipped: list[Path], root: str | Path) -> str:
    """One line naming what a directory scan left out, worst-case truncated."""
    root = Path(root)
    names = [str(p.relative_to(root)) if p.is_relative_to(root) else str(p) for p in skipped]
    shown = ", ".join(names[:3]) + (", …" if len(names) > 3 else "")
    return (
        f"skipped {len(names)} hidden or lock file(s) under {root} ({shown}); "
        "pass a file to --docs directly to include it."
    )


def _unique_ids(documents: list[Document], files: list[Path], root: Path) -> list[Document]:
    """Give every document an id of its own.

    A document is known by its file name without the extension, so `2023/msa.md`
    and `2024/msa.md` were both `msa`, and their chunks shared ids. Only files
    whose names collide are renamed, to their path under the scanned folder:
    a corpus without a collision keeps its ids and its `corpus_sha256`.
    """
    ids = [d.id for d in documents]
    paths = [f.relative_to(root).as_posix() for f in files]
    # Paths are unique, so this ends; a second round is needed only when a stem
    # equals another file's path (`x.md.txt` next to a renamed `x.md`).
    while True:
        counts = Counter(ids)
        clashing = [i for i, doc_id in enumerate(ids) if counts[doc_id] > 1 and doc_id != paths[i]]
        if not clashing:
            break
        for i in clashing:
            ids[i] = paths[i]
    return [
        d if d.id == doc_id else d.model_copy(update={"id": doc_id})
        for d, doc_id in zip(documents, ids, strict=True)
    ]


def load_documents(docs_path: str | Path, skipped: list[Path] | None = None) -> list[Document]:
    """Load a single file or every supported file in a directory (recursive).

    A directory scan leaves out hidden files and folders and office lock files,
    and appends each one to `skipped` when a list is given, so the caller can say
    so. A file named explicitly is always loaded. Files that share a name are
    told apart by their path under the folder.
    """
    path = Path(docs_path)
    left_out: list[Path] = []
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = []
        for candidate in sorted(path.rglob("*")):
            if not candidate.is_file() or candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            (left_out if _not_meant_for_evaluation(candidate, path) else files).append(candidate)
        if skipped is not None:
            skipped.extend(left_out)
    else:
        raise InputNotFoundError(f"docs path not found: {path}")
    if not files:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        detail = f"; {describe_skipped(left_out, path)}" if left_out else ""
        raise InputError(
            f"no supported documents found under {path} (supported: {supported}){detail}"
        )

    documents: list[Document] = []
    for file in files:
        try:
            documents.append(get_loader(file.suffix).load(file))
        except Exception as exc:  # a raw parser traceback does not name the file
            raise DocumentLoadError(f"could not read {file}: {type(exc).__name__}: {exc}") from exc
    return _unique_ids(documents, files, path if path.is_dir() else path.parent)
