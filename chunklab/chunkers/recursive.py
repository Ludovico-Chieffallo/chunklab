"""Recursive splitter (wraps langchain-text-splitters), mapped back to char spans."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunklab.chunkers.base import build_chunk
from chunklab.models import Chunk, Document
from chunklab.text_utils import count_tokens


def map_pieces_to_spans(text: str, pieces: list[str]) -> list[tuple[int, int]]:
    """Locate each piece in `text` in order, searching forward from the start of
    the previous match (pieces may overlap, so we cannot search from its end)."""
    spans: list[tuple[int, int]] = []
    search_from = 0
    for piece in pieces:
        idx = text.find(piece, search_from)
        if idx == -1:
            # The splitter strips whitespace; retry from the beginning as a fallback.
            idx = text.find(piece)
            if idx == -1:
                raise ValueError("could not map chunk back to source text")
        spans.append((idx, idx + len(piece)))
        search_from = idx + 1
    return spans


class RecursiveChunker:
    name = "recursive"

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=count_tokens,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
        )

    def chunk(self, document: Document) -> list[Chunk]:
        pieces = self._splitter.split_text(document.text)
        spans = map_pieces_to_spans(document.text, pieces)
        return [build_chunk(document, self.name, i, s) for i, s in enumerate(spans)]
