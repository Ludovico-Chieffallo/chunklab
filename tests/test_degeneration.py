"""The anti-degeneration test (roadmap 2.2): the ranking metric must not be
gameable by chunk size.

`whole_document` (one chunk per document) maximizes containment-based recall
by construction. Under plain `recall_at_k` it wins — that IS the bias. Under
`balanced` it must lose to any reasonable strategy.
"""

from pathlib import Path

import pytest

from chunklab.config import StrategyConfig, default_config, load_questions
from chunklab.runner import run_evaluation

TEST_DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def inputs(request):
    from chunklab.loaders.text import TextLoader

    doc = TextLoader().load(TEST_DATA / "employee_handbook.md")
    questions = load_questions(TEST_DATA / "questions.example.yaml")
    return [doc], questions


def _config(ranking_metric: str):
    config = default_config()
    config.embedding.backend = "fake"
    config.eval.ranking_metric = ranking_metric  # type: ignore[assignment]
    config.strategies = [
        StrategyConfig(name="whole_document"),
        StrategyConfig(name="structure", params={"max_tokens": 800}),
        StrategyConfig(name="recursive", params={"chunk_size": 512, "overlap": 64}),
    ]
    return config


def test_degenerate_strategy_wins_on_raw_recall(inputs):
    """Documents the bias: containment recall alone rewards the biggest chunk."""
    docs, questions = inputs
    report = run_evaluation(docs, questions, _config("recall_at_k"))
    by_name = {r.strategy: r for r in report.strategy_results}
    assert by_name["whole_document"].recall_at_k >= by_name["structure"].recall_at_k


def test_degenerate_strategy_does_not_win(inputs):
    """The fix: under `balanced`, the degenerate strategy must not rank first."""
    docs, questions = inputs
    report = run_evaluation(docs, questions, _config("balanced"))
    assert report.strategy_results[0].strategy != "whole_document"
    by_name = {r.strategy: r for r in report.strategy_results}
    # And its penalty must be material, not a rounding artifact.
    assert by_name["whole_document"].balanced_score < by_name["whole_document"].recall_at_k - 0.10


def test_whole_document_hidden_from_listing():
    from chunklab.chunkers.registry import available_strategies

    assert "whole_document" not in available_strategies()
