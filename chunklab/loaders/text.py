"""Loader for .txt and .md files."""

import codecs
from pathlib import Path

from chunklab.loaders.base import make_doc_id
from chunklab.loaders.markdown_elements import extract_markdown_elements
from chunklab.models import Document


def read_text(path: Path) -> tuple[str, str]:
    """Return (text, encoding actually used).

    utf-8-sig decodes plain UTF-8 identically *and* strips a byte-order mark. The
    BOM matters: a leading U+FEFF is invisible in an editor but detaches the first
    heading from the start of the line, so a document saved by a Windows editor
    silently lost every element it had. cp1252 then covers the common
    non-UTF-8 Western encoding rather than failing the whole corpus.
    """
    raw = path.read_bytes()
    had_bom = raw.startswith(codecs.BOM_UTF8)
    try:
        return raw.decode("utf-8-sig"), "utf-8-bom" if had_bom else "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("cp1252"), "cp1252"
    except UnicodeDecodeError:
        # Nearly unreachable - cp1252 rejects only a few bytes - but reading the
        # corpus with replacements beats refusing to read it at all.
        return raw.decode("utf-8", errors="replace"), "utf-8/replace"


class TextLoader:
    def load(self, path: Path) -> Document:
        text, encoding = read_text(path)
        elements = extract_markdown_elements(text) if path.suffix.lower() == ".md" else []
        return Document(
            id=make_doc_id(path),
            source_path=str(path),
            text=text,
            elements=elements,
            metadata={
                "format": path.suffix.lstrip(".").lower() or "txt",
                "encoding": encoding,
            },
        )
