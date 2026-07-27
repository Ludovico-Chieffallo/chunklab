"""Strategy name -> chunker factory, including user-registered plugins."""

from chunklab.chunkers.base import Chunker

BUILTIN_STRATEGIES: dict[str, str] = {
    "fixed": "Fixed-size token windows with overlap (chunk_size=512, overlap=64)",
    "recursive": "Recursive splitting on paragraph/sentence separators (chunk_size=512)",
    "semantic": "Embedding-based boundaries with a min-size floor (min_tokens=200)",
    "semantic_no_floor": "Semantic without the floor - demonstrates the fragment trap",
    "structure": "Heading-aware sections, sub-split when oversized (max_tokens=800)",
}

# Valid in configs but not advertised: test/diagnostic strategies.
HIDDEN_STRATEGIES = {"whole_document"}


def available_strategies() -> dict[str, str]:
    """Built-in strategies plus anything registered or published by a plugin."""
    from chunklab.plugins import registered_chunkers

    strategies = dict(BUILTIN_STRATEGIES)
    for name, plugin in registered_chunkers().items():
        strategies[name] = plugin.description or "(registered plugin)"
    return strategies


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
    if name == "whole_document":
        from chunklab.chunkers.whole_document import WholeDocumentChunker

        return WholeDocumentChunker(**params)

    from chunklab.plugins import build_plugin_chunker, registered_chunkers

    if name in registered_chunkers():
        return build_plugin_chunker(name, params, embedder)

    known = ", ".join(sorted(available_strategies()))
    raise ValueError(f"unknown strategy '{name}'; available: {known}")
