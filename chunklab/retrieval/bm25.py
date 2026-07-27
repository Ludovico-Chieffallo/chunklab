"""Okapi BM25 lexical retrieval.

Implemented here rather than pulled in as a dependency: the whole scorer is
thirty lines, and owning it means owning the tokenizer, which is what decides
whether the retriever works outside English at all.

BM25 is the honest counterweight to dense retrieval in this tool. Embeddings
generalise across wording but blur exact tokens — identifiers, clause numbers,
error codes, product names — which are precisely what people search contracts
and API references for. Comparing chunking strategies under only one retriever
measures how well chunking suits *that* retriever.
"""

import math
from collections import Counter

from chunklab.models import Chunk, RetrievedChunk
from chunklab.retrieval.tokenize import tokenize

#: Term-frequency saturation. Above this, repeating a term adds almost nothing.
K1 = 1.5
#: Length normalization strength: 0 ignores chunk length, 1 fully normalizes.
B = 0.75


class BM25Retriever:
    """Classic Okapi BM25 over an in-memory chunk index."""

    def __init__(self, chunks: list[Chunk], k1: float = K1, b: float = B) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        self._docs = [tokenize(chunk.text) for chunk in chunks]
        self._lengths = [len(doc) for doc in self._docs]
        self._avg_length = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        self._freqs = [Counter(doc) for doc in self._docs]

        doc_freq: Counter[str] = Counter()
        for freq in self._freqs:
            doc_freq.update(freq.keys())

        n = len(self._docs)
        # Robertson/Sparck-Jones idf with the +1 smoothing that keeps it positive
        # for terms present in every chunk.
        self._idf = {
            term: math.log(1.0 + (n - count + 0.5) / (count + 0.5))
            for term, count in doc_freq.items()
        }

    def _score(self, index: int, query_terms: list[str]) -> float:
        freq = self._freqs[index]
        length = self._lengths[index]
        norm = (
            self.k1 * (1 - self.b + self.b * length / self._avg_length)
            if self._avg_length
            else self.k1
        )
        total = 0.0
        for term in query_terms:
            count = freq.get(term)
            if not count:
                continue
            total += self._idf.get(term, 0.0) * (count * (self.k1 + 1)) / (count + norm)
        return total

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        query_terms = tokenize(query)
        scores = [self._score(i, query_terms) for i in range(len(self.chunks))]
        # Sort by score, breaking ties by chunk order so runs stay deterministic.
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [
            RetrievedChunk(chunk=self.chunks[i], score=float(scores[i]), rank=rank + 1)
            for rank, i in enumerate(order[: min(top_k, len(self.chunks))])
        ]
