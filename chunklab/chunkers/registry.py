"""Strategy name -> chunker factory, including user-registered plugins."""

import inspect

from chunklab.chunkers.base import Chunker
from chunklab.errors import InputError

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


_SEMANTIC = {"semantic", "semantic_no_floor"}

#: Arguments chunklab passes itself. They are not the user's to set in `params`.
_INJECTED = {"embedder", "strategy_name"}


def _builtin_factory(name: str):
    """The class behind a built-in strategy, or None when `name` is not one."""
    if name == "fixed":
        from chunklab.chunkers.fixed import FixedChunker

        return FixedChunker
    if name == "recursive":
        from chunklab.chunkers.recursive import RecursiveChunker

        return RecursiveChunker
    if name == "structure":
        from chunklab.chunkers.structure import StructureChunker

        return StructureChunker
    if name in _SEMANTIC:
        from chunklab.chunkers.semantic import SemanticChunker

        return SemanticChunker
    if name == "whole_document":
        from chunklab.chunkers.whole_document import WholeDocumentChunker

        return WholeDocumentChunker
    return None


def _factory(name: str):
    factory = _builtin_factory(name)
    if factory is not None:
        return factory

    from chunklab.plugins import registered_chunkers

    plugin = registered_chunkers().get(name)
    if plugin is not None:
        return plugin.factory
    known = ", ".join(sorted(available_strategies()))
    raise InputError(f"unknown strategy '{name}'; available: {known}")


def check_strategy_params(name: str, params: dict) -> None:
    """Refuse `params` the strategy's constructor does not take, before any work.

    `params` comes straight from the user's config. A misspelled key used to
    surface as a TypeError traceback when that strategy's turn came, after the
    documents and the embedding model had loaded; this names the key and lists
    the real ones.
    """
    try:
        signature = inspect.signature(_factory(name))
    except (TypeError, ValueError):  # builtins and C callables: nothing to check
        return
    parameters = signature.parameters.values()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
        return
    settable = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    accepted = [p.name for p in parameters if p.kind in settable and p.name not in _INJECTED]

    unknown = [key for key in params if key not in accepted]
    if unknown:
        raise InputError(
            f"strategy '{name}' has no parameter {', '.join(repr(k) for k in unknown)}; "
            f"accepted: {', '.join(accepted) or 'none'}"
        )
    missing = [
        p.name
        for p in parameters
        if p.kind in settable
        and p.default is inspect.Parameter.empty
        and p.name not in _INJECTED
        and p.name not in params
    ]
    if missing:
        raise InputError(f"strategy '{name}' needs parameter {', '.join(repr(k) for k in missing)}")


def make_chunker(name: str, params: dict, embedder=None) -> Chunker:
    factory = _factory(name)
    check_strategy_params(name, params)
    if name in _SEMANTIC and embedder is None:
        raise ValueError(f"strategy '{name}' requires an embedder")

    try:
        if name in _SEMANTIC:
            return factory(embedder, strategy_name=name, **params)
        if _builtin_factory(name) is None:
            from chunklab.plugins import build_plugin_chunker

            return build_plugin_chunker(name, params, embedder)
        return factory(**params)
    except InputError:
        raise
    except ValueError as exc:
        # Constructors check their values (overlap below chunk_size, ...). Those
        # values came from the user's config, so the failure is theirs to fix.
        raise InputError(f"strategy '{name}': {exc}") from exc
