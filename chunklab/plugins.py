"""Registering your own chunking strategy.

chunklab exists to answer "which way of splitting my documents retrieves best".
That question is only worth asking if you can put *your* splitter in the
comparison — the one your pipeline actually uses, with its own rules about
tables, headers or clause boundaries. Without this, chunklab can only rank the
five strategies it happens to ship.

Two ways in, both of which produce an ordinary strategy name usable anywhere a
built-in name is:

    # in your own code, before calling evaluate()
    from chunklab.plugins import register_chunker
    register_chunker("my_splitter", MySplitter, "Splits on our clause markers")

    # or from an installed package, discovered automatically
    [project.entry-points."chunklab.chunkers"]
    my_splitter = "my_package.chunkers:MySplitter"

A factory is anything callable that returns an object with `.chunk(document)`.
It receives the strategy's `params` as keyword arguments, plus `embedder=` only
if its signature asks for one — so a splitter that needs no model does not have
to accept an argument it will not use.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass

from chunklab.chunkers.base import Chunker

ENTRY_POINT_GROUP = "chunklab.chunkers"


@dataclass(frozen=True)
class ChunkerPlugin:
    name: str
    factory: Callable[..., Chunker]
    description: str


_REGISTRY: dict[str, ChunkerPlugin] = {}
_entry_points_loaded = False


def register_chunker(
    name: str, factory: Callable[..., Chunker], description: str = "", replace: bool = False
) -> None:
    """Make `name` usable as a strategy.

    Refuses to shadow a built-in or an already registered name unless `replace`
    is set: a silent override would change what a config file means without any
    sign of it in the report.
    """
    from chunklab.chunkers.registry import BUILTIN_STRATEGIES, HIDDEN_STRATEGIES

    if not name or not name.strip():
        raise ValueError("a chunker name cannot be empty")
    if name in BUILTIN_STRATEGIES or name in HIDDEN_STRATEGIES:
        raise ValueError(f"'{name}' is a built-in strategy; choose another name")
    if name in _REGISTRY and not replace:
        raise ValueError(f"'{name}' is already registered; pass replace=True to override")
    if not callable(factory):
        raise TypeError(f"factory for '{name}' is not callable")

    _REGISTRY[name] = ChunkerPlugin(name=name, factory=factory, description=description)


def unregister_chunker(name: str) -> None:
    """Remove a registered chunker. Mostly for tests."""
    _REGISTRY.pop(name, None)


def _load_entry_points() -> None:
    """Discover chunkers published by installed packages, once."""
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    _entry_points_loaded = True

    from importlib.metadata import entry_points

    for entry in entry_points(group=ENTRY_POINT_GROUP):
        try:
            factory = entry.load()
        except Exception as exc:  # a broken plugin must not break the whole run
            import warnings

            warnings.warn(
                f"chunklab plugin '{entry.name}' failed to load and was skipped: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if entry.name not in _REGISTRY:
            _REGISTRY[entry.name] = ChunkerPlugin(
                name=entry.name,
                factory=factory,
                description=getattr(factory, "__doc__", "") or "",
            )


def registered_chunkers() -> dict[str, ChunkerPlugin]:
    _load_entry_points()
    return dict(_REGISTRY)


def build_plugin_chunker(name: str, params: dict, embedder=None) -> Chunker:
    """Instantiate a registered chunker, passing `embedder` only if it wants one."""
    plugin = registered_chunkers()[name]
    factory = plugin.factory

    if _wants_embedder(factory):
        chunker = factory(embedder=embedder, **params)
    else:
        chunker = factory(**params)

    if not hasattr(chunker, "chunk"):
        raise TypeError(f"chunker '{name}' has no .chunk(document) method")
    # The strategy name ends up in every chunk id and in the report, so it has to
    # be the name the user configured, whatever the class calls itself.
    try:
        chunker.name = name
    except AttributeError:
        pass
    return chunker


def _wants_embedder(factory: Callable) -> bool:
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):  # builtins and C callables
        return False
    parameters = signature.parameters
    if "embedder" in parameters:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
