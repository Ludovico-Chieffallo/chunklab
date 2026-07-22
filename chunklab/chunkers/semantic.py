"""Semantic chunker with a minimum-size floor (the fix for the fragment trap).

Algorithm (see spec §4.1):
  1. sentence-split, 2. embed sentences, 3. breakpoints where consecutive cosine
  distance exceeds the `breakpoint_percentile`-th percentile, 4. group,
  5. floor-merge groups under `min_tokens`, 6. ceiling-split groups over
  `max_tokens` at sentence boundaries.

`min_tokens=0` disables the floor — the `semantic_no_floor` variant that
demonstrates the trap.
"""

import numpy as np

from chunklab.chunkers.base import build_chunk
from chunklab.embeddings.base import Embedder
from chunklab.models import Chunk, Document
from chunklab.text_utils import count_tokens, sentence_spans


class SemanticChunker:
    name = "semantic"

    def __init__(
        self,
        embedder: Embedder,
        breakpoint_percentile: float = 95,
        min_tokens: int = 200,
        max_tokens: int = 1000,
        strategy_name: str | None = None,
    ) -> None:
        self.embedder = embedder
        self.breakpoint_percentile = breakpoint_percentile
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        if strategy_name:
            self.name = strategy_name

    def chunk(self, document: Document) -> list[Chunk]:
        spans = sentence_spans(document.text)
        if not spans:
            return []
        sentences = [document.text[s:e] for s, e in spans]
        tokens = [count_tokens(s) for s in sentences]

        def span_tokens(group: tuple[int, int]) -> int:
            # Count on the real span text: separators between sentences add tokens
            # that per-sentence sums would miss.
            return count_tokens(document.text[spans[group[0]][0] : spans[group[1]][1]])

        groups = self._group_by_breakpoints(sentences)
        groups = self._floor_merge(groups, tokens)
        groups = self._ceiling_split(groups, tokens, span_tokens)

        chunks: list[Chunk] = []
        for first, last in groups:
            span = (spans[first][0], spans[last][1])
            chunks.append(build_chunk(document, self.name, len(chunks), span))
        return chunks

    def _group_by_breakpoints(self, sentences: list[str]) -> list[tuple[int, int]]:
        """Return groups as (first_sentence_idx, last_sentence_idx) inclusive."""
        n = len(sentences)
        if n == 1:
            return [(0, 0)]
        vectors = self.embedder.embed(sentences)
        # vectors are unit-normalized; cosine distance = 1 - dot
        distances = 1.0 - np.sum(vectors[:-1] * vectors[1:], axis=1)
        threshold = float(np.percentile(distances, self.breakpoint_percentile))
        groups: list[tuple[int, int]] = []
        start = 0
        for i, dist in enumerate(distances):
            if dist > threshold:
                groups.append((start, i))
                start = i + 1
        groups.append((start, n - 1))
        return groups

    def _group_tokens(self, group: tuple[int, int], tokens: list[int]) -> int:
        return sum(tokens[group[0] : group[1] + 1])

    def _floor_merge(
        self, groups: list[tuple[int, int]], tokens: list[int]
    ) -> list[tuple[int, int]]:
        if self.min_tokens <= 0:
            return groups
        groups = list(groups)
        while len(groups) > 1:
            sizes = [self._group_tokens(g, tokens) for g in groups]
            tiny = [i for i, s in enumerate(sizes) if s < self.min_tokens]
            if not tiny:
                break
            i = tiny[0]
            # Merge with the neighbor that yields the smaller resulting chunk.
            left = sizes[i - 1] if i > 0 else None
            right = sizes[i + 1] if i < len(groups) - 1 else None
            if right is None or (left is not None and left <= right):
                groups[i - 1 : i + 1] = [(groups[i - 1][0], groups[i][1])]
            else:
                groups[i : i + 2] = [(groups[i][0], groups[i + 1][1])]
        return groups

    def _ceiling_split(
        self, groups: list[tuple[int, int]], tokens: list[int], span_tokens
    ) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        stack = list(reversed(groups))
        while stack:
            first, last = stack.pop()
            total = span_tokens((first, last))
            if total <= self.max_tokens or first == last:
                result.append((first, last))
                continue
            # Split at the sentence boundary closest to the token midpoint.
            acc, split = 0, first
            for i in range(first, last):
                acc += tokens[i]
                split = i
                if acc >= total / 2:
                    break
            stack.append((split + 1, last))
            stack.append((first, split))
        return result
