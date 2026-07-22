"""Strategy name -> chunker factory."""

from chunklab.chunkers.base import Chunker


def available_strategies() -> dict[str, str]:
    return {
        "fixed": "Fixed-size token windows with overlap (chunk_size=512, overlap=64)",
        "recursive": "Recursive splitting on paragraph/sentence separators (chunk_size=512)",
        "semantic": "Embedding-based boundaries with a min-size floor (min_tokens=200)",
        "semantic_no_floor": "Semantic without the floor - demonstrates the fragment trap",
        "structure": "Heading-aware sections, sub-split when oversized (max_tokens=800)",
    }


def make_chunker(name: str, params: dict, embedder=None) -> Chunker:
    if name == "fixed":
        from chunklab.chunkers.fixed import FixedChunker

        return FixedChunker(**params)
    if name == "recursive":
        from chunklab.chunkers.recursive import RecursiveChunker

        return RecursiveChunker(**params)
    if name == "structure":
        from chunklab.chunkers.structure import StructureChunker

        return StructureChunker(**params)
    if name in {"semantic", "semantic_no_floor"}:
        from chunklab.chunkers.semantic import SemanticChunker

        if embedder is None:
            raise ValueError(f"strategy '{name}' requires an embedder")
        return SemanticChunker(embedder, strategy_name=name, **params)
    raise ValueError(f"unknown strategy '{name}'")
