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


class DocxLoader:
    def load(self, path: Path) -> Document:
        import docx
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        d = docx.Document(str(path))
        parts: list[str] = []
        elements: list[Element] = []
        offset = 0

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
            metadata={"format": "docx"},
        )
