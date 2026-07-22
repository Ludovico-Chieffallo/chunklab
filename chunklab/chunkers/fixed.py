"""Fixed-size token windows with overlap."""

from chunklab.chunkers.base import build_chunk
from chunklab.models import Chunk, Document
from chunklab.text_utils import token_spans


class FixedChunker:
    name = "fixed"

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        spans = token_spans(document.text)
        if not spans:
            return []
        step = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        i = 0
        while i < len(spans):
            window = spans[i : i + self.chunk_size]
            start, end = window[0][0], window[-1][1]
            chunks.append(build_chunk(document, self.name, len(chunks), (start, end)))
            if i + self.chunk_size >= len(spans):
                break
            i += step
        return chunks
