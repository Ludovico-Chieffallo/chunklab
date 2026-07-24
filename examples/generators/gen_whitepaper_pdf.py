"""Regenerate examples/corpus/whitepaper.pdf from whitepaper_text.txt.

The whitepaper is deliberately heading-free flowing prose: it exists to stress
chunkers that depend on document structure. Rendering is line-by-line (no
silent text drops) with hyphen-preserving wrapping so gold snippets survive
the PDF -> text roundtrip verbatim.

Usage: python gen_whitepaper_pdf.py
"""

import textwrap
from pathlib import Path

import fitz

HERE = Path(__file__).parent
SOURCE = HERE / "whitepaper_text.txt"
TARGET = HERE.parent / "corpus" / "whitepaper.pdf"

W, H, M = 595, 842, 56  # A4 points, margin
LINE_H = 14.5


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    paragraphs = [p.strip().replace("\n", " ") for p in text.split("\n\n") if p.strip()]

    doc = fitz.open()
    page = doc.new_page(width=W, height=H)
    y = M + 10
    for i, para in enumerate(paragraphs):
        fontsize = 14 if i == 0 else 10.5
        lines = textwrap.wrap(para, width=92 if i else 70, break_on_hyphens=False)
        for line in lines:
            if y > H - M:
                page = doc.new_page(width=W, height=H)
                y = M + 10
            page.insert_text((M, y), line, fontsize=fontsize, fontname="helv")
            y += LINE_H
        y += 8

    doc.save(TARGET)
    print(f"wrote {TARGET} ({len(doc)} pages)")


if __name__ == "__main__":
    main()
