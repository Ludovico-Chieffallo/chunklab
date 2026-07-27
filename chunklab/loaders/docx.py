"""DOCX loader via python-docx.

Builds Document.text by concatenating paragraphs and tables in document order,
tracking char offsets as we go (offsets are into the text we build, so they are
exact by construction).
"""

import re
from pathlib import Path

from chunklab.loaders.base import make_doc_id
from chunklab.models import Document, Element

_HEADING_STYLE_RE = re.compile(r"heading\s*(\d)", re.IGNORECASE)

#: Longest a visually-formatted heading is allowed to be, in words.
_MAX_INFERRED_HEADING_WORDS = 12

#: Trailing punctuation that marks a sentence rather than a heading.
_SENTENCE_TAIL = (".", ",", ";", "?", "!")


def _is_bold(run) -> bool:
    """True when the run is bold directly or through its character style."""
    if run.bold is not None:
        return run.bold
    style = getattr(run, "style", None)
    font = getattr(style, "font", None)
    return bool(getattr(font, "bold", False))


def _inferred_heading_level(paragraph) -> int | None:
    """Heading level for a paragraph formatted as one, or None.

    Documents converted from HTML or PDF routinely carry no heading *styles* at
    all - the Microsoft annual report used for testing has 961 paragraphs, every
    one of them `Normal (Web)` - while still being visually structured. Without
    this, `structure` chunking silently degrades to packing paragraphs up to its
    token cap, which is not what the user asked for and was not reported.

    The level is a guess (all-caps reads as more prominent), so it is recorded
    but should not be trusted as a real outline depth.
    """
    text = paragraph.text.strip()
    if not text or len(text.split()) > _MAX_INFERRED_HEADING_WORDS:
        return None
    if text.endswith(_SENTENCE_TAIL):
        return None  # a short bold sentence is emphasis, not a heading
    runs = [run for run in paragraph.runs if run.text.strip()]
    if not runs or not all(_is_bold(run) for run in runs):
        return None
    return 2 if text.isupper() else 3


class DocxLoader:
    def load(self, path: Path) -> Document:
        import docx
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        d = docx.Document(str(path))
        parts: list[str] = []
        elements: list[Element] = []
        offset = 0
        inferred_headings = 0

        def append(text: str, el_type: str, level: int | None = None) -> None:
            nonlocal offset
            start = offset
            parts.append(text + "\n\n")
            offset += len(text) + 2
            elements.append(
                Element(type=el_type, text=text, char_span=(start, start + len(text)), level=level)
            )

        for item in d.iter_inner_content():
            if isinstance(item, Paragraph):
                text = item.text.strip()
                if not text:
                    continue
                m = _HEADING_STYLE_RE.match(item.style.name or "")
                if m:
                    level = int(m.group(1))
                    append("#" * level + " " + text, "heading", level)
                    continue
                level = _inferred_heading_level(item)
                if level is not None:
                    inferred_headings += 1
                    append("#" * level + " " + text, "heading", level)
                else:
                    append(text, "paragraph")
            elif isinstance(item, Table):
                rows = []
                for row in item.rows:
                    cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                    rows.append("| " + " | ".join(cells) + " |")
                if rows:
                    append("\n".join(rows), "table")

        return Document(
            id=make_doc_id(path),
            source_path=str(path),
            text="".join(parts),
            elements=elements,
            metadata={"format": "docx", "inferred_headings": inferred_headings},
        )
