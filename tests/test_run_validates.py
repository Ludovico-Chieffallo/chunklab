"""`run` checks the question set before spending a run on it.

A gold snippet that is not in the corpus scores zero for every strategy, and nothing
said so: with two questions and one bad snippet, recall was 0.50 in every cell and the
report carried no warning. The README calls that a mistake that "looks like a
chunking problem and is not", and only `chunklab validate`, which nobody has to run,
caught it. The evaluation now starts with the same checks and stops on an error,
before the embedding model loads. `--skip-validation` scores the set as it is.
"""

import pytest

from chunklab.config import Config, default_config
from chunklab.errors import InvalidQuestionSet
from chunklab.models import Document, Question
from chunklab.runner import run_evaluation

TEXT = (
    "# Billing\n\n"
    "Invoices are issued on the first business day of each month and are payable "
    "within 45 days of issue. Late payments accrue interest at 1.5% per month.\n"
)
DOC = Document(id="billing", source_path="billing.md", text=TEXT)
GOOD = Question(
    id="q1", query="When are invoices payable?", gold_snippets=["payable within 45 days of issue"]
)
MISSING = Question(
    id="q2",
    query="Who approves refunds?",
    gold_snippets=["refunds are approved by the finance team lead"],
)


def _fake() -> Config:
    """The default strategies, on the deterministic offline embedder."""
    config = default_config()
    config.embedding.backend = "fake"
    return config


QUESTIONS_YAML = (
    "questions:\n"
    "  - id: q1\n"
    '    query: "When are invoices payable?"\n'
    '    gold_snippets: ["payable within 45 days of issue"]\n'
    "  - id: q2\n"
    '    query: "Who approves refunds?"\n'
    '    gold_snippets: ["refunds are approved by the finance team lead"]\n'
)


@pytest.fixture()
def corpus(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "billing.md").write_text(TEXT, encoding="utf-8")
    return docs


@pytest.fixture()
def questions(tmp_path):
    path = tmp_path / "questions.yaml"
    path.write_text(QUESTIONS_YAML, encoding="utf-8")
    return path


# --- the evaluation -------------------------------------------------------------------


def test_a_snippet_missing_from_the_corpus_stops_the_run_before_the_model_loads(monkeypatch):
    def no_model(*args, **kwargs):
        raise AssertionError("the embedding model was loaded")

    monkeypatch.setattr("chunklab.runner.make_embedder", no_model)

    with pytest.raises(InvalidQuestionSet) as caught:
        run_evaluation([DOC], [GOOD, MISSING], _fake())

    assert [issue.question_id for issue in caught.value.errors] == ["q2"]
    assert "q2" in str(caught.value)


def test_skip_validation_scores_the_set_as_it_is():
    """The silent result this replaces: half the recall, in every cell."""
    report = run_evaluation([DOC], [GOOD, MISSING], _fake(), skip_validation=True)

    assert {r.recall_at_k for r in report.strategy_results} == {0.5}


def test_warnings_alone_do_not_stop_the_run():
    unannotated = Question(id="q3", query="Is there a late fee?")

    report = run_evaluation([DOC], [GOOD, unannotated], _fake())

    assert report.strategy_results


# --- the CLI --------------------------------------------------------------------------


def test_run_lists_the_errors_and_writes_nothing(tmp_path, cli, corpus, questions, fake_config):
    out = tmp_path / "out"

    result = cli(
        "run", "--docs", corpus, "--questions", questions, "--config", fake_config, "--out", out
    )

    assert result.exit_code == 2, result.output
    assert "Traceback" not in result.output
    assert "q2 (not_found)" in result.output
    assert "--skip-validation" in result.output
    assert not out.exists()


def test_run_with_skip_validation_writes_the_reports(tmp_path, cli, corpus, questions, fake_config):
    out = tmp_path / "out"

    result = cli(
        "run",
        "--docs",
        corpus,
        "--questions",
        questions,
        "--config",
        fake_config,
        "--out",
        out,
        "--skip-validation",
    )

    assert result.exit_code == 0, result.output
    assert (out / "report.json").is_file()


def test_check_validates_too(tmp_path, cli, corpus, questions, fake_config):
    """A baseline recorded from a broken set would gate CI on the mistake."""
    baseline = tmp_path / "baseline.json"

    result = cli(
        "check",
        "--docs",
        corpus,
        "--questions",
        questions,
        "--config",
        fake_config,
        "--baseline",
        baseline,
        "--update-baseline",
    )

    assert result.exit_code == 2, result.output
    assert "q2 (not_found)" in result.output
    assert not baseline.exists()
