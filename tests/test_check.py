"""The CI regression gate (roadmap phase 8).

The design question is what makes a build fail. A tool that refuses to name a
winner on noise must not fail a build on noise either, so the default gate is
the same paired bootstrap the recommendation uses.

The other half is when a build may pass. Until 0.5.0, a run that could not be
compared with its baseline - other questions, so no pairing - printed "No
regression detected." in green and exited 0: the gate stayed off, silently, until
someone re-recorded the baseline. Now it fails closed.
"""

import pytest

import chunklab
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
        corpus_summary={
            "questions_sha256": "same-questions",
            "chunklab_version": chunklab.__version__,
            "top_k": 5,
            "fuzzy_threshold": 0.9,
            "corpus_sha256": "same-corpus",
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "embedding_model_revision": "rev-a",
        },
        strategy_results=[
            StrategyResult(
                strategy=strategy,
                retriever=retriever,
                config={"chunk_size": 512, "overlap": 64},
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


def test_changed_question_set_is_not_comparable():
    """It used to pass, green, with the statistical gate silently off."""
    baseline = _report(_scores(30))
    current = _report(_scores(20))
    current.corpus_summary["questions_sha256"] = "different-questions"

    outcome = compare(baseline, current)

    assert outcome.status == "not_comparable" and not outcome.ok
    assert outcome.ci95 is None
    assert any("question set changed" in reason for reason in outcome.not_comparable)


def test_allow_unpaired_compares_the_raw_numbers():
    baseline = _report(_scores(30))
    current = _report(_scores(20))
    current.corpus_summary["questions_sha256"] = "different-questions"

    outcome = compare(baseline, current, allow_unpaired=True)

    assert outcome.status == "pass"
    assert outcome.ci95 is None
    assert any("no statistical gate" in note for note in outcome.notes)


def test_unpaired_still_honours_max_drop():
    baseline = _report(_scores(30))
    current = _report(_scores(10))
    current.corpus_summary["questions_sha256"] = "different-questions"

    outcome = compare(baseline, current, max_drop=0.05, allow_unpaired=True)

    assert outcome.status == "regression"


@pytest.mark.parametrize(
    ("field", "value", "said"),
    [
        ("chunklab_version", "0.0.1", "chunklab 0.0.1"),
        ("top_k", 10, "top 10"),
        ("fuzzy_threshold", 0.8, "fuzzy threshold"),
    ],
)
def test_a_change_to_the_measurement_is_not_comparable(field, value, said):
    """A different version can score the same retrieval differently (0.5.0 does),
    and recall at another k, or under another threshold, is another measure."""
    baseline = _report(_scores(30))
    baseline.corpus_summary[field] = value

    outcome = compare(baseline, _report(_scores(30)))

    assert outcome.status == "not_comparable"
    assert any(said in reason for reason in outcome.not_comparable), outcome.not_comparable


def test_a_baseline_cell_missing_from_the_run_is_not_comparable():
    baseline = _report(_scores(30), retriever="hybrid")

    outcome = compare(baseline, _report(_scores(30), retriever="dense"))

    assert outcome.status == "not_comparable"
    assert outcome.current_recall is None
    assert any("'recursive + hybrid' is not in this run" in r for r in outcome.not_comparable)


def test_what_check_exists_to_catch_is_compared_and_named():
    """A new corpus, model revision or chunk size is the change under test: it is
    gated, and said, so a result can be read."""
    baseline = _report(_scores(30))
    current = _report(_scores(30))
    current.corpus_summary["corpus_sha256"] = "edited-corpus"
    current.corpus_summary["embedding_model_revision"] = "rev-b"
    current.strategy_results[0].config = {"chunk_size": 256, "overlap": 64}

    outcome = compare(baseline, current)

    assert outcome.status == "pass" and outcome.comparable
    notes = " ".join(outcome.notes)
    assert "corpus changed" in notes
    assert "rev-a" in notes and "rev-b" in notes
    assert "512" in notes and "256" in notes


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


# --- the command ----------------------------------------------------------------------

DOC = (
    "# Billing\n\n"
    "Invoices are issued on the first business day of each month and are payable "
    "within 45 days of issue. Late payments accrue interest at 1.5% per month.\n"
)
QUESTION = (
    "  - id: {qid}\n"
    '    query: "When are invoices payable?"\n'
    '    gold_snippets: ["payable within 45 days of issue"]\n'
)


@pytest.fixture()
def project(tmp_path, cli, fake_config):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "billing.md").write_text(DOC, encoding="utf-8")
    questions = tmp_path / "questions.yaml"
    questions.write_text("questions:\n" + QUESTION.format(qid="q1"), encoding="utf-8")
    baseline = tmp_path / "baseline.json"

    def check(*extra):
        args = ["--docs", docs, "--questions", questions, "--config", fake_config]
        return cli("check", *args, "--baseline", baseline, *extra)

    assert check("--update-baseline").exit_code == 0
    return check, questions


def test_an_unchanged_run_passes(project):
    check, _ = project

    result = check()

    assert result.exit_code == 0, result.output
    assert "No regression detected." in result.output


def test_a_run_on_other_questions_exits_3_and_says_why(project):
    check, questions = project
    questions.write_text("questions:\n" + QUESTION.format(qid="q1-renamed"), encoding="utf-8")

    result = check()

    assert result.exit_code == 3, result.output
    assert "Not comparable" in result.output and "question set changed" in result.output
    assert "--allow-unpaired" in result.output
    assert "No regression detected." not in result.output


def test_allow_unpaired_restores_the_raw_comparison(project):
    check, questions = project
    questions.write_text("questions:\n" + QUESTION.format(qid="q1-renamed"), encoding="utf-8")

    result = check("--allow-unpaired")

    assert result.exit_code == 0, result.output
    assert "Unpaired:" in result.output
