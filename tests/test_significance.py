"""Tests for bootstrap CIs and the tie-declaring recommendation (roadmap 2.3)."""

from chunklab.config import default_config
from chunklab.eval.significance import (
    bootstrap_mean_ci,
    estimate_questions_to_separate,
    paired_bootstrap_diff_ci,
)
from chunklab.models import ChunkHealth, QuestionResult, StrategyResult
from chunklab.runner import _build_recommendation


def test_bootstrap_ci_covers_mean():
    values = [0.0, 0.5, 1.0, 1.0, 0.5, 0.0, 1.0, 0.5]
    lo, hi = bootstrap_mean_ci(values, resamples=2000, seed=1)
    mean = sum(values) / len(values)
    assert lo <= mean <= hi
    assert 0.0 <= lo < hi <= 1.0


def test_bootstrap_ci_deterministic_with_seed():
    values = [0.2, 0.4, 0.6, 0.8]
    assert bootstrap_mean_ci(values, seed=7) == bootstrap_mean_ci(values, seed=7)


def test_paired_diff_ci_identical_scores_is_zero():
    a = [1.0, 0.5, 0.0, 1.0]
    assert paired_bootstrap_diff_ci(a, list(a), resamples=500, seed=0) == (0.0, 0.0)


def test_paired_diff_ci_detects_clear_gap():
    a = [1.0] * 30
    b = [0.0] * 30
    lo, hi = paired_bootstrap_diff_ci(a, b, resamples=500, seed=0)
    assert lo > 0.5  # clearly separated


def test_estimate_questions_to_separate():
    assert estimate_questions_to_separate(20, 0.0, (-0.1, 0.1)) is None
    n = estimate_questions_to_separate(20, 0.02, (-0.08, 0.12))
    assert n is not None and n > 20


def _strategy_result(name: str, per_q_scores: list[float]) -> StrategyResult:
    health = ChunkHealth(
        num_chunks=10, tokens_min=100, tokens_median=400, tokens_mean=400,
        tokens_max=800, pct_tiny=0.0, pct_oversized=0.0, boundary_health=1.0,
    )
    per_question = [
        QuestionResult(
            question_id=f"q{i}", strategy=name, retrieved=[], hit=s > 0,
            gold_found_count=int(s * 4), gold_total=4,
        )
        for i, s in enumerate(per_q_scores)
    ]
    mean = sum(per_q_scores) / len(per_q_scores)
    return StrategyResult(
        strategy=name, recall_at_k=mean, hit_rate_at_k=mean, mrr=mean,
        precision_at_k=mean, chunk_health=health, per_question=per_question,
    )


def test_recommendation_declares_tie():
    scores = [1.0, 0.75, 0.5, 0.25, 1.0, 0.5, 0.75, 1.0]
    ranked = [_strategy_result("fixed", scores), _strategy_result("recursive", list(scores))]
    config = default_config()
    text = _build_recommendation(ranked, config, num_scored=len(scores))
    assert "statistically indistinguishable" in text
    assert "Use " not in text  # no winner is recommended


def test_recommendation_names_winner_when_separated():
    ranked = [
        _strategy_result("structure", [1.0] * 30),
        _strategy_result("fixed", [0.0] * 30),
    ]
    config = default_config()
    text = _build_recommendation(ranked, config, num_scored=30)
    assert "Use STRUCTURE" in text
