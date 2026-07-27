"""Hybrid retrieval by Reciprocal Rank Fusion.

RRF combines rankings rather than scores, which is what makes it usable here:
BM25 scores are unbounded sums of idf weights and cosine similarities live in
[-1, 1], so any weighted sum of the two would need per-corpus calibration that
chunklab has no way to do honestly. Ranks need none.

    score(chunk) = sum over retrievers of 1 / (RRF_K + rank)

`RRF_K = 60` is the value from Cormack, Clarke & Buettcher (SIGIR 2009), where
it was found insensitive across collections. It damps the top ranks so a single
retriever's confident-but-wrong first hit cannot dominate the fusion.
"""

from chunklab.models import RetrievedChunk
from chunklab.retrieval.base import Retriever

RRF_K = 60

#: How deep each retriever is asked to go before fusing. Fusing only the final
#: top-k would discard chunks that one retriever ranked 6th and the other 7th -
#: exactly the agreement RRF exists to reward.
FUSION_DEPTH_MULTIPLIER = 4
MIN_FUSION_DEPTH = 20


class HybridRetriever:
    """Fuses several retrievers' rankings; order of `retrievers` is irrelevant."""

    def __init__(self, retrievers: list[Retriever], rrf_k: int = RRF_K) -> None:
        if not retrievers:
            raise ValueError("hybrid retrieval needs at least one retriever")
        self.retrievers = retrievers
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        depth = max(MIN_FUSION_DEPTH, top_k * FUSION_DEPTH_MULTIPLIER)

        fused: dict[str, float] = {}
        seen: dict[str, RetrievedChunk] = {}
        best_rank: dict[str, int] = {}

        for retriever in self.retrievers:
            for hit in retriever.retrieve(query, depth):
                key = hit.chunk.id
                fused[key] = fused.get(key, 0.0) + 1.0 / (self.rrf_k + hit.rank)
                best_rank[key] = min(best_rank.get(key, hit.rank), hit.rank)
                seen.setdefault(key, hit)

        # Exact ties are common: two retrievers that swap ranks 1 and 3 give both
        # chunks the same fused score. Breaking that by insertion order made the
        # ranking depend on the order `retrievers` was passed in, so the same
        # corpus could rank differently for no reason. Both tiebreakers below are
        # intrinsic to the chunk, so the fusion is order-independent.
        order = sorted(fused, key=lambda key: (-fused[key], best_rank[key], key))
        return [
            RetrievedChunk(chunk=seen[key].chunk, score=fused[key], rank=rank + 1)
            for rank, key in enumerate(order[:top_k])
        ]
