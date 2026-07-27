"""The CI regression gate (roadmap phase 8).

The design question is what makes a build fail. A tool that refuses to name a
winner on noise must not fail a build on noise either, so the default gate is
the same paired bootstrap the recommendation uses.
"""

import pytest

from chunklab.check import compare, load_baseline
from chunklab.models import ChunkHealth, EvalReport, QuestionResult, StrategyResult

HEALTH = ChunkHealth(
    num_chunks=10,
    tokens_min=100,
    tokens_median=400,
    tokens_mean=400,
    tokens_max=500,
    pct_tiny=0.0,
    pct_oversized=0.0,
    boundary_health=1.0,
)


def _report(scores: dict[str, float], strategy: str = "recursive", retriever: str = "dense"):
    per_question = [
        QuestionResult(
            question_id=qid,
            strategy=strategy,
            retrieved=[],
            hit=score > 0,
            gold_found_count=int(round(score)),
            gold_total=1,
        )
        for qid, score in scores.items()
    ]
    recall = sum(scores.values()) / len(scores)
    return EvalReport(
        corpus_summary={"questions_sha256": "same-questions"},
        strategy_results=[
            StrategyResult(
                strategy=strategy,
                retriever=retriever,
                recall_at_k=recall,
                hit_rate_at_k=recall,
                mrr=recall,
                precision_at_k=recall / 5,
                chunk_health=HEALTH,
                per_question=per_question,
            )
        ],
    )


def _scores(hits: int, total: int = 40) -> dict[str, float]:
    return {f"q{i}": (1.0 if i < hits else 0.0) for i in range(total)}


def test_identical_runs_pass():
    outcome = compare(_report(_scores(30)), _report(_scores(30)))

    assert outcome.ok
    assert outcome.difference == 0.0
    assert outcome.comparable


def test_large_real_regression_fails():
    outcome = compare(_report(_scores(34)), _report(_scores(14)))

    assert not outcome.ok
    assert outcome.ci95 is not None and outcome.ci95[1] < 0
    assert "larger than sampling noise" in outcome.failures[0]


def test_small_drop_within_noise_does_not_fail():
    """One question out of 40 flipping must not break a build."""
    outcome = compare(_report(_scores(30)), _report(_scores(29)))

    assert outcome.ok, outcome.failures
    assert outcome.difference < 0


def test_improvement_never_fails():
    outcome = compare(_report(_scores(20)), _report(_scores(35)))
    assert outcome.ok


def test_max_drop_is_an_opt_in_blunt_floor():
    baseline, current = _report(_scores(30)), _report(_scores(28))

    assert compare(baseline, current).ok, "default gate should tolerate noise"
    strict = compare(baseline, current, max_drop=0.01)
    assert not strict.ok
    assert "--max-drop" in strict.failures[0]


def test_changed_question_set_is_not_paired():
    baseline = _report(_scores(30))
    current = _report(_scores(20))
    current.corpus_summary["questions_sha256"] = "different-questions"

    outcome = compare(baseline, current)

    assert not outcome.comparable
    assert outcome.ci95 is None
    assert any("question set changed" in n for n in outcome.notes)
    assert outcome.ok, "an unpaired comparison must not fail the build by itself"


def test_unpaired_still_honours_max_drop():
    baseline = _report(_scores(30))
    current = _report(_scores(10))
    current.corpus_summary["questions_sha256"] = "different-questions"

    outcome = compare(baseline, current, max_drop=0.05)

    assert not outcome.ok


def test_tracks_the_baselines_winner_by_default():
    baseline = _report(_scores(30), strategy="structure")
    current = _report(_scores(30), strategy="structure")

    assert compare(baseline, current).strategy == "structure"


def test_named_strategy_must_exist():
    with pytest.raises(ValueError, match="no strategy matching"):
        compare(_report(_scores(30)), _report(_scores(30)), strategy="nonexistent")


def test_retriever_is_matched_not_just_the_strategy():
    baseline = _report(_scores(30), retriever="hybrid")
    current = _report(_scores(30), retriever="hybrid")

    outcome = compare(baseline, current)

    assert outcome.retriever == "hybrid"


def test_baseline_from_an_incompatible_schema_is_refused(tmp_path):
    stale = _report(_scores(30))
    stale.schema_version = "0.9"
    path = tmp_path / "baseline.json"
    path.write_text(stale.model_dump_json(), encoding="utf-8")

    with pytest.raises(ValueError, match="regenerate the baseline"):
        load_baseline(path)


def test_baseline_roundtrips_through_json(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text(_report(_scores(30)).model_dump_json(), encoding="utf-8")

    loaded = load_baseline(path)

    assert loaded.strategy_results[0].strategy == "recursive"
