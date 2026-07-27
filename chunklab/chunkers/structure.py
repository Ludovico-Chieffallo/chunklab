"""Heading-aware chunker: one chunk per section, sub-splitting oversized sections."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunklab.chunkers.base import build_chunk
from chunklab.chunkers.recursive import map_pieces_to_spans
from chunklab.models import Chunk, Document
from chunklab.text_utils import count_tokens


class StructureChunker:
    name = "structure"

    def __init__(self, max_tokens: int = 800) -> None:
        self.max_tokens = max_tokens
        self._sub_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_tokens,
            chunk_overlap=0,
            length_function=count_tokens,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
        )

    @staticmethod
    def _split_depth(headings: list) -> int:
        """Deepest heading level that still counts as a section boundary.

        H1-H3 is the right cut for a conventional outline, but converters do not
        produce conventional outlines: pymupdf4llm assigned level 5 to 261 of the
        268 headings in a real 10-K, so a fixed `<= 3` recognised three headings
        in a 66k-token filing and silently packed the rest to `max_tokens`.

        When the bulk of the headings sit deeper than H3, the outline is simply
        shifted, so follow where the headings actually are.
        """
        levels = [el.level or 1 for el in headings]
        most_common = max(set(levels), key=levels.count)
        return max(3, most_common)

    def _section_spans(self, document: Document) -> list[tuple[int, int]]:
        headings = [el for el in document.elements if el.type == "heading" and el.level is not None]
        if not headings:
            return [(0, len(document.text))]
        depth = self._split_depth(headings)
        # Split at the structural levels; deeper headings stay inside their section.
        starts = [el.char_span[0] for el in headings if (el.level or 1) <= depth]
        if not starts:
            return [(0, len(document.text))]
        if starts[0] != 0:
            starts = [0, *starts]
        spans = [
            (s, e) for s, e in zip(starts, [*starts[1:], len(document.text)], strict=False) if e > s
        ]
        # A parent heading immediately followed by a subheading yields a
        # heading-only span; merge it into the following section instead of
        # emitting a tiny heading chunk.
        merged: list[tuple[int, int]] = []
        i = 0
        while i < len(spans):
            start, end = spans[i]
            body = document.text[start:end].strip()
            first_line_only = "\n" not in body
            if first_line_only and i + 1 < len(spans):
                spans[i + 1] = (start, spans[i + 1][1])
            else:
                merged.append((start, end))
            i += 1
        return merged

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        for start, end in self._section_spans(document):
            section = document.text[start:end]
            if not section.strip():
                continue
            if count_tokens(section) <= self.max_tokens:
                spans = [(start, end)]
            else:
                pieces = self._sub_splitter.split_text(section)
                spans = [(start + s, start + e) for s, e in map_pieces_to_spans(section, pieces)]
            for span in spans:
                chunks.append(build_chunk(document, self.name, len(chunks), span))
        return chunks
