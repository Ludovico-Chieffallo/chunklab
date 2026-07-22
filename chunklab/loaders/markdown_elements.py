"""Shared extraction of structural elements (headings, tables) from markdown-ish text.

Used by the text/md loader and by the PDF loader (pymupdf4llm emits markdown).
Char spans are offsets into the exact text passed in.
"""

import re

from chunklab.models import Element

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


def extract_markdown_elements(text: str) -> list[Element]:
    elements: list[Element] = []

    for m in _HEADING_RE.finditer(text):
        elements.append(
            Element(
                type="heading",
                text=m.group(2),
                char_span=(m.start(), m.end()),
                level=len(m.group(1)),
            )
        )

    # Tables: maximal runs of consecutive '|...|' lines (>= 2 lines).
    lines = text.splitlines(keepends=True)
    offset = 0
    run_start: int | None = None
    run_end = 0
    run_len = 0

    def flush() -> None:
        nonlocal run_start, run_len
        if run_start is not None and run_len >= 2:
            elements.append(
                Element(
                    type="table",
                    text=text[run_start:run_end].rstrip("\n"),
                    char_span=(run_start, run_end),
                )
            )
        run_start = None
        run_len = 0

    for line in lines:
        if _TABLE_ROW_RE.match(line.rstrip("\n")):
            if run_start is None:
                run_start = offset
                run_len = 0
            run_end = offset + len(line)
            run_len += 1
        else:
            flush()
        offset += len(line)
    flush()

    elements.sort(key=lambda e: e.char_span[0])
    return elements
