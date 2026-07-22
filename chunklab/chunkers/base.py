"""Chunker protocol and shared helpers for building Chunk objects."""

from typing import Protocol, runtime_checkable

from chunklab.models import Chunk, Document
from chunklab.text_utils import count_tokens


@runtime_checkable
class Chunker(Protocol):
    name: str

    def chunk(self, document: Document) -> list[Chunk]: ...


def section_path_at(document: Document, char_pos: int) -> list[str]:
    """Heading trail (H1 > H2 > ...) in effect at `char_pos`."""
    stack: list[tuple[int, str]] = []  # (level, text)
    for el in document.elements:
        if el.type != "heading" or el.char_span[0] > char_pos:
            continue
        level = el.level or 1
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, el.text))
    return [t for _, t in stack]


def span_contains_table(document: Document, span: tuple[int, int]) -> bool:
    return any(
        el.type == "table" and el.char_span[0] < span[1] and el.char_span[1] > span[0]
        for el in document.elements
    )


def build_chunk(document: Document, strategy: str, index: int, span: tuple[int, int]) -> Chunk:
    text = document.text[span[0] : span[1]]
    return Chunk(
        id=f"{document.id}:{strategy}:{index}",
        doc_id=document.id,
        text=text,
        token_count=count_tokens(text),
        char_span=span,
        strategy=strategy,
        contains_table=span_contains_table(document, span),
        section_path=section_path_at(document, span[0]),
    )
