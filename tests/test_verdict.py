"""The verdict as data, for the program that has to act on it.

`recommendation` is English. A CI job that wants to know whether the run named a
winner, or which strategies it ruled out, had to parse sentences - and
docs/schema.md described the field as "naming the winner", which most runs do
not. `verdict` holds the same decision, axis by axis, and must never disagree
with the prose.
"""

import json

from chunklab.config import default_config
from chunklab.models import (
    ChunkHealth,
    Document,
    EvalReport,
    Question,
    QuestionResult,
    StrategyResult,
)
from chunklab.runner import _recommend, run_evaluation

HEALTH = ChunkHealth(
    num_chunks=10,
    tokens_min=100,
    tokens_median=400,
    tokens_mean=400,
    tokens_max=800,
    pct_tiny=0.0,
    pct_oversized=0.0,
    boundary_health=1.0,
)


def _cell(strategy: str, scores: list[float], retriever: str = "dense") -> StrategyResult:
    mean = sum(scores) / len(scores)
    return StrategyResult(
        strategy=strategy,
        retriever=retriever,
        recall_at_k=mean,
        hit_rate_at_k=mean,
        mrr=mean,
        precision_at_k=mean,
        retrieved_tokens_at_k=2000.0,
        balanced_score=mean,
        chunk_health=HEALTH,
        per_question=[
            QuestionResult(
                question_id=f"q{i}",
                strategy=strategy,
                retrieved=[],
                hit=s > 0,
                gold_found_count=int(s),
                gold_total=1,
            )
            for i, s in enumerate(scores)
        ],
    )


LEADER = [1.0] * 30 + [0.0] * 10
RIVAL = [0.0, 0.0] + LEADER[2:30] + [1.0, 1.0] + LEADER[32:]
CLOSE = LEADER[:2] + [0.0, 0.0] + LEADER[4:]
FAR = LEADER[:4] + [0.0] * 10 + LEADER[14:]


def test_an_undecided_axis_lists_what_is_in_play_and_what_is_out():
    ranked = [
        _cell("structure", LEADER),
        _cell("semantic", RIVAL),
        _cell("recursive", CLOSE),
        _cell("fixed", FAR),
    ]

    text, verdict = _recommend(ranked, default_config(), num_scored=40)

    chunking = verdict.chunking
    assert chunking.axis == "chunking"
    assert not chunking.decided and chunking.winner is None
    assert chunking.leader == "structure"
    assert chunking.in_play == ["structure", "semantic", "recursive"]
    assert chunking.ruled_out == ["fixed"]
    assert abs(sum(chunking.probability_best.values()) - 1.0) < 1e-9
    assert verdict.retrieval is None
    # The prose says the same thing.
    assert "3 cannot be ruled out ('structure', 'semantic', 'recursive')" in text
    assert "'fixed' (0%) can" in text


def test_a_decided_axis_names_its_winner_and_margin():
    ranked = [_cell("structure", [1.0] * 30), _cell("fixed", [0.0] * 30)]

    text, verdict = _recommend(ranked, default_config(), num_scored=30)

    chunking = verdict.chunking
    assert chunking.decided and chunking.winner == "structure"
    assert chunking.probability_best["structure"] >= 0.90
    assert chunking.margin.runner_up == "fixed"
    assert chunking.margin.difference == 1.0
    assert chunking.questions_to_separate is None
    assert "Use STRUCTURE" in text


def test_a_matrix_has_one_verdict_per_axis_and_names_what_each_was_held_at():
    hits = lambda n: [1.0] * n + [0.0] * (20 - n)  # noqa: E731
    ranked = sorted(
        [
            _cell("recursive", hits(13), "hybrid"),
            _cell("recursive", hits(12), "bm25"),
            _cell("semantic", hits(4), "hybrid"),
            _cell("semantic", hits(4), "bm25"),
        ],
        key=lambda r: -r.recall_at_k,
    )

    _text, verdict = _recommend(ranked, default_config(), num_scored=20)

    assert verdict.chunking.held == "hybrid"
    assert verdict.chunking.decided and verdict.chunking.winner == "recursive"
    assert verdict.retrieval is not None and verdict.retrieval.axis == "retrieval"
    assert verdict.retrieval.held == "recursive"
    assert set(verdict.retrieval.probability_best) == {"hybrid", "bm25"}


def test_the_report_carries_the_verdict_through_json():
    config = default_config()
    config.embedding.backend = "fake"
    document = Document(
        id="billing",
        source_path="billing.md",
        text="# Billing\n\nInvoices are payable within 45 days of issue.\n",
    )
    question = Question(id="q1", query="?", gold_snippets=["payable within 45 days of issue"])

    report = run_evaluation([document], [question], config)
    again = EvalReport.model_validate_json(report.model_dump_json())

    assert again.verdict == report.verdict
    assert report.verdict.chunking.leader == report.strategy_results[0].strategy
    assert "verdict" in json.loads(report.model_dump_json())
