"""The Gradio demo, which the README advertises and nobody had ever executed.

Its imports resolving was the only thing known about it. These tests build the
interface and run a real evaluation through it, so `chunklab demo` cannot rot
into a command that fails the moment someone clicks the button.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from app import EXAMPLE_QUESTIONS, TABLE_HEADERS, parse_questions, run_eval  # noqa: E402

HANDBOOK = Path(__file__).parent / "data" / "employee_handbook.md"


# --- question parsing (no gradio needed) ---------------------------------------


def test_question_and_gold_are_split_on_the_separator():
    questions = parse_questions("How is overtime paid? :: Overtime is paid at 1.5x")

    assert len(questions) == 1
    assert questions[0].query == "How is overtime paid?"
    assert questions[0].gold_snippets == ["Overtime is paid at 1.5x"]


def test_a_line_without_a_separator_becomes_an_unscored_question():
    questions = parse_questions("How is overtime paid?")

    assert questions[0].gold_snippets == []


def test_blank_lines_are_ignored():
    assert len(parse_questions("a :: b\n\n\nc :: d\n")) == 2


def test_gold_containing_the_separator_keeps_everything_after_the_first():
    questions = parse_questions("Query? :: gold :: with colons")

    assert questions[0].gold_snippets == ["gold :: with colons"]


def test_the_prefilled_example_parses_and_is_scorable():
    """The textbox ships with this text; if it does not parse, the first click fails."""
    questions = parse_questions(EXAMPLE_QUESTIONS)

    assert len(questions) == 2
    assert all(q.gold_snippets for q in questions)


# --- guard rails ----------------------------------------------------------------


def test_missing_file_is_a_clear_error():
    with pytest.raises(ValueError, match="Upload a document"):
        run_eval(None, EXAMPLE_QUESTIONS, ["fixed"])


def test_questions_without_gold_are_refused():
    with pytest.raises(ValueError, match="gold snippet"):
        run_eval(str(HANDBOOK), "How is overtime paid?", ["fixed"])


# --- a real evaluation through the demo path ------------------------------------


QUESTIONS = "How is overtime compensated? :: Overtime is paid at 1.5x the regular hourly rate"


@pytest.fixture(autouse=True)
def offline_embeddings(monkeypatch):
    """Run the demo's own code path without downloading a model.

    The demo calls `default_config()`, which asks for the real embedder. What is
    under test here is the demo's plumbing — parsing, table shape, report file —
    not retrieval quality, so a deterministic stand-in keeps these fast enough to
    run on every commit instead of being skipped into irrelevance.
    """
    import app as demo_module
    from chunklab.config import default_config as real_default_config

    def fake_backend_config():
        config = real_default_config()
        config.embedding.backend = "fake"
        return config

    monkeypatch.setattr(demo_module, "default_config", fake_backend_config)


@pytest.fixture
def demo_result():
    return run_eval(str(HANDBOOK), QUESTIONS, ["fixed", "recursive"])


def test_evaluation_returns_a_recommendation(demo_result):
    recommendation, _table, _path = demo_result
    assert recommendation.strip()


def test_table_rows_match_the_declared_headers(demo_result):
    """The Dataframe is built with TABLE_HEADERS; a row of a different width
    silently misaligns every column."""
    _rec, (headers, rows), _path = demo_result

    assert len(headers) == len(TABLE_HEADERS)
    assert rows and all(len(row) == len(headers) for row in rows)
    assert {row[0] for row in rows} == {"fixed", "recursive"}


def test_headers_carry_the_retrieval_cutoff(demo_result):
    _rec, (headers, _rows), _path = demo_result

    assert "recall@5" in headers
    assert "{k}" not in " ".join(headers), "placeholder left unformatted"


def test_html_report_is_written(demo_result):
    _rec, _table, path = demo_result

    report = Path(path)
    assert report.exists() and report.stat().st_size > 0
    assert "<html" in report.read_text(encoding="utf-8").lower()


def test_selecting_a_subset_of_strategies_is_honoured():
    _rec, (_headers, rows), _path = run_eval(str(HANDBOOK), QUESTIONS, ["fixed"])

    assert {row[0] for row in rows} == {"fixed"}


# --- the interface itself (needs gradio) ----------------------------------------


def test_interface_builds():
    """Regression guard: gradio's API changes between majors, and this is the only
    place chunklab touches it."""
    pytest.importorskip("gradio")
    from app import build_app

    assert build_app() is not None
