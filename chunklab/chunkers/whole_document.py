"""Degenerate chunker: one chunk per document.

Deliberately hidden from `chunklab strategies`. It exists to test that the
ranking metric cannot be gamed by chunk size: recall-by-containment always
favors the biggest possible chunk, so a sound ranking metric must not let
this strategy win. See tests/test_degeneration.py and docs/metrics.md.
"""

from chunklab.chunkers.base import build_chunk
from chunklab.models import Chunk, Document


class WholeDocumentChunker:
    name = "whole_document"

    def chunk(self, document: Document) -> list[Chunk]:
        if not document.text.strip():
            return []
        return [build_chunk(document, self.name, 0, (0, len(document.text)))]
