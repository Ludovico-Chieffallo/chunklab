"""Build the requested retrievers over one strategy's chunks, sharing indexes."""

from chunklab.embeddings.base import Embedder
from chunklab.models import Chunk
from chunklab.retrieval.base import Retriever

RETRIEVAL_MODES = ("dense", "bm25", "hybrid")


def make_retrievers(
    modes: list[str], chunks: list[Chunk], embedder: Embedder
) -> dict[str, Retriever]:
    """Retriever per mode. `hybrid` reuses the dense and BM25 indexes it fuses,
    so asking for all three costs no more than asking for the two."""
    unknown = [mode for mode in modes if mode not in RETRIEVAL_MODES]
    if unknown:
        raise ValueError(
            f"unknown retrieval mode(s) {unknown}; available: {', '.join(RETRIEVAL_MODES)}"
        )

    needs_dense = any(mode in {"dense", "hybrid"} for mode in modes)
    needs_lexical = any(mode in {"bm25", "hybrid"} for mode in modes)

    dense = None
    lexical = None
    if needs_dense:
        from chunklab.retrieval.dense import DenseRetriever

        dense = DenseRetriever(chunks, embedder)
    if needs_lexical:
        from chunklab.retrieval.bm25 import BM25Retriever

        lexical = BM25Retriever(chunks)

    built: dict[str, Retriever] = {}
    for mode in modes:
        if mode == "dense":
            built[mode] = dense
        elif mode == "bm25":
            built[mode] = lexical
        else:
            from chunklab.retrieval.hybrid import HybridRetriever

            built[mode] = HybridRetriever([dense, lexical])
    return built
