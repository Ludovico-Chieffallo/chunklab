"""The console table is what every user actually reads, and it had no tests.

Everything else in this project can be right while this renders the wrong
number, hides the winner, or silently drops a warning — and nobody would know,
because it is the one output no other test touches.
"""

import pytest
from rich.console import Console

from chunklab.models import ChunkHealth, EvalReport, StrategyResult
from chunklab.report.console import print_report


def _health(**overrides) -> ChunkHealth:
    values = dict(
        num_chunks=42,
        tokens_min=80,
        tokens_median=410.0,
        tokens_mean=400.0,
        tokens_max=512,
        pct_tiny=0.05,
        pct_oversized=0.0,
        boundary_health=0.97,
    )
    values.update(overrides)
    return ChunkHealth(**values)


def _result(strategy: str, recall: float, retriever: str = "dense", **overrides) -> StrategyResult:
    values = dict(
        strategy=strategy,
        retriever=retriever,
        recall_at_k=recall,
        hit_rate_at_k=recall,
        mrr=recall * 0.8,
        precision_at_k=recall / 5,
        retrieved_tokens_at_k=2000.0,
        balanced_score=recall,
        chunk_health=_health(),
    )
    values.update(overrides)
    return StrategyResult(**values)


def _report(results, **summary) -> EvalReport:
    corpus_summary = {
        "num_documents": 3,
        "num_scored_questions": 40,
        "top_k": 5,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "ranking_metric": "balanced",
    }
    corpus_summary.update(summary)
    return EvalReport(
        corpus_summary=corpus_summary,
        strategy_results=results,
        recommendation="Use RECURSIVE chunking. It gave the best retrieval.",
    )


def render(report: EvalReport, width: int = 200) -> str:
    console = Console(record=True, width=width, force_terminal=False)
    print_report(report, console)
    return console.export_text()


def test_header_states_the_corpus_and_model():
    text = render(_report([_result("recursive", 0.82)]))

    assert "3 document(s)" in text
    assert "40 scored questions" in text
    assert "top_k=5" in text
    assert "BAAI/bge-small-en-v1.5" in text


def test_every_strategy_appears_with_its_numbers():
    text = render(_report([_result("recursive", 0.82), _result("fixed", 0.71)]))

    assert "recursive" in text and "fixed" in text
    assert "0.82" in text and "0.71" in text


def test_the_winner_is_marked():
    """The first row is the recommendation; without the marker the table is just data."""
    text = render(_report([_result("recursive", 0.82), _result("fixed", 0.71)]))

    winner_line = next(line for line in text.splitlines() if "recursive" in line)
    assert "▶" in winner_line

    runner_up = next(line for line in text.splitlines() if "fixed" in line)
    assert "▶" not in runner_up


def test_recommendation_and_warnings_are_printed():
    report = _report([_result("recursive", 0.82)])
    report.warnings = ["only 4 scored questions: differences are unlikely to be meaningful"]

    text = render(report)

    assert "Use RECURSIVE chunking" in text
    assert "only 4 scored questions" in text
    assert "Warning" in text


def test_the_model_caveat_is_always_shown():
    """Measured: swapping the embedding model changed the winner. A ranking that
    does not name its model is not a result."""
    text = render(_report([_result("recursive", 0.82)]))

    assert "BAAI/bge-small-en-v1.5" in text
    assert "run with the one you deploy" in text


# --- the balanced column follows the configured ranking metric ------------------


def test_balanced_column_shown_when_it_is_the_ranking_metric():
    text = render(_report([_result("recursive", 0.82)], ranking_metric="balanced"))
    assert "balanced" in text


def test_balanced_column_hidden_for_other_metrics():
    text = render(_report([_result("recursive", 0.82)], ranking_metric="recall_at_k"))
    assert "balanced" not in text.lower().split("recall")[0]


# --- the retriever column only when there is something to compare ---------------


def test_retriever_column_absent_with_a_single_retriever():
    text = render(_report([_result("recursive", 0.82), _result("fixed", 0.71)]))

    header = text.splitlines()[3]
    assert "retriever" not in header


def test_retriever_column_present_in_a_matrix():
    results = [
        _result("recursive", 0.92, retriever="hybrid"),
        _result("recursive", 0.82, retriever="dense"),
    ]

    text = render(_report(results, retrieval_modes=["dense", "hybrid"]))

    assert "retriever" in text
    assert "hybrid" in text and "dense" in text


# --- shapes that must not raise -------------------------------------------------


@pytest.mark.parametrize(
    ("label", "results"),
    [
        ("nessuna strategia", []),
        ("valori a zero", [_result("fixed", 0.0, retrieved_tokens_at_k=0.0)]),
        ("recall perfetta", [_result("fixed", 1.0)]),
    ],
)
def test_degenerate_reports_still_render(label, results):
    text = render(_report(results))
    assert "ChunkLab" in text, label


def test_missing_summary_fields_do_not_crash():
    """A report from an older schema, or a hand-built one, must still print."""
    report = EvalReport(corpus_summary={}, strategy_results=[_result("fixed", 0.5)])

    text = render(report)

    assert "ChunkLab" in text


def test_narrow_terminal_does_not_raise():
    render(_report([_result("recursive", 0.82)]), width=40)
