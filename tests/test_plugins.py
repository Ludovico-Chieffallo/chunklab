"""Registering your own chunking strategy (roadmap phase 8).

chunklab can only rank the strategies it can see. If your pipeline uses a
splitter of your own, being unable to put it in the comparison makes the
comparison beside the point.
"""

import pytest

from chunklab.chunkers.base import build_chunk
from chunklab.chunkers.registry import available_strategies, make_chunker
from chunklab.config import default_config
from chunklab.models import Document, Question
from chunklab.plugins import register_chunker, unregister_chunker
from chunklab.runner import run_evaluation

TEXT = (
    "Invoices are payable within thirty days.\n\n"
    "Support acknowledges a production outage within one hour.\n\n"
    "The rate limit is 300 requests per minute on the Starter plan.\n\n"
    "Audit entries are immutable and readable for 400 days.\n\n"
) * 3


class ParagraphChunker:
    """Splits on blank lines - the sort of thing a user already has."""

    name = "paragraph"

    def __init__(self, min_chars: int = 1) -> None:
        self.min_chars = min_chars

    def chunk(self, document: Document):
        chunks = []
        offset = 0
        for block in document.text.split("\n\n"):
            start = document.text.index(block, offset) if block else offset
            end = start + len(block)
            offset = end
            if len(block.strip()) >= self.min_chars:
                chunks.append(build_chunk(document, self.name, len(chunks), (start, end)))
        return chunks


class EmbedderAwareChunker:
    name = "needs_embedder"

    def __init__(self, embedder=None, size: int = 2) -> None:
        self.embedder = embedder
        self.size = size

    def chunk(self, document: Document):
        return [build_chunk(document, self.name, 0, (0, len(document.text)))]


@pytest.fixture
def registered():
    names: list[str] = []

    def _register(name, factory, description=""):
        register_chunker(name, factory, description)
        names.append(name)
        return name

    yield _register
    for name in names:
        unregister_chunker(name)


def _document() -> Document:
    return Document(id="doc", source_path="d.md", text=TEXT, elements=[], metadata={})


def test_registered_chunker_is_usable_as_a_strategy(registered):
    registered("paragraph", ParagraphChunker, "Splits on blank lines")

    chunker = make_chunker("paragraph", {})
    chunks = chunker.chunk(_document())

    assert len(chunks) > 1
    assert all(c.strategy == "paragraph" for c in chunks)


def test_registered_chunker_is_listed(registered):
    registered("paragraph", ParagraphChunker, "Splits on blank lines")

    assert available_strategies()["paragraph"] == "Splits on blank lines"


def test_params_reach_the_factory(registered):
    registered("paragraph", ParagraphChunker)

    chunker = make_chunker("paragraph", {"min_chars": 10_000})

    assert chunker.chunk(_document()) == []


def test_embedder_is_passed_only_when_wanted(registered, fake_embedder):
    registered("needs_embedder", EmbedderAwareChunker)
    registered("paragraph", ParagraphChunker)

    aware = make_chunker("needs_embedder", {}, embedder=fake_embedder)
    plain = make_chunker("paragraph", {}, embedder=fake_embedder)

    assert aware.embedder is fake_embedder
    assert not hasattr(plain, "embedder"), "a splitter that wants no model must not get one"


def test_strategy_name_follows_the_registration(registered):
    """The configured name ends up in chunk ids and the report, so it must win."""
    registered("my_alias", ParagraphChunker)

    chunks = make_chunker("my_alias", {}).chunk(_document())

    assert chunks[0].strategy == "my_alias"
    assert chunks[0].id.startswith("doc:my_alias:")


def test_builtin_names_cannot_be_shadowed():
    with pytest.raises(ValueError, match="built-in"):
        register_chunker("fixed", ParagraphChunker)


def test_hidden_names_cannot_be_shadowed():
    with pytest.raises(ValueError, match="built-in"):
        register_chunker("whole_document", ParagraphChunker)


def test_duplicate_registration_needs_replace(registered):
    registered("paragraph", ParagraphChunker)

    with pytest.raises(ValueError, match="already registered"):
        register_chunker("paragraph", ParagraphChunker)

    register_chunker("paragraph", ParagraphChunker, replace=True)


def test_empty_name_is_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        register_chunker("   ", ParagraphChunker)


def test_non_callable_factory_is_rejected():
    with pytest.raises(TypeError, match="not callable"):
        register_chunker("bad", "not a factory")


def test_unknown_strategy_lists_what_is_available():
    with pytest.raises(ValueError, match="available:"):
        make_chunker("no_such_strategy", {})


def test_object_without_chunk_method_is_rejected(registered):
    registered("broken", lambda **kwargs: object())

    with pytest.raises(TypeError, match="no .chunk"):
        make_chunker("broken", {})


def test_plugin_competes_with_the_builtins_in_a_real_run(registered):
    registered("paragraph", ParagraphChunker, "Splits on blank lines")

    config = default_config()
    config.embedding.backend = "fake"
    config.strategies = [
        s for s in config.strategies if s.name in {"fixed", "recursive"}
    ]
    from chunklab.config import StrategyConfig

    config.strategies.append(StrategyConfig(name="paragraph", params={}))

    questions = [
        Question(
            id="q1",
            query="when are invoices payable?",
            gold_snippets=["Invoices are payable within thirty days"],
        )
    ]
    report = run_evaluation([_document()], questions, config)

    assert "paragraph" in {r.strategy for r in report.strategy_results}


def test_config_accepts_a_registered_strategy(registered):
    from chunklab.config import StrategyConfig

    registered("paragraph", ParagraphChunker)

    assert StrategyConfig(name="paragraph").name == "paragraph"


def test_config_still_rejects_unknown_strategies():
    from pydantic import ValidationError

    from chunklab.config import StrategyConfig

    with pytest.raises(ValidationError):
        StrategyConfig(name="definitely_not_registered")
