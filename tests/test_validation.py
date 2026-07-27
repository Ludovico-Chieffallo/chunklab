"""Tests for `chunklab validate` (roadmap 3.1).

Acceptance criterion: on a question set with planted errors, the command finds
all of them and proposes the verbatim correction for drifted snippets.
"""

from pathlib import Path

import pytest

from chunklab.config import load_questions
from chunklab.loaders.registry import load_documents
from chunklab.models import Document, Question
from chunklab.validation import normalize_with_map, validate_questions

EXAMPLES = Path(__file__).parent.parent / "examples"

DOC_TEXT = (
    "# Handbook\n\n"
    "Overtime is paid at 1.5x the regular hourly rate for all hours worked beyond forty.\n\n"
    "Employees must give written notice at least 30 days prior to termination.\n"
)


@pytest.fixture()
def doc() -> Document:
    return Document(id="handbook", source_path="handbook.md", text=DOC_TEXT)


def test_normalize_with_map_offsets_point_at_source():
    text = "  Hello   World\n"
    norm, offsets = normalize_with_map(text)
    assert norm == "hello world"
    assert text[offsets[0]] == "H"
    assert text[offsets[norm.index("w")]] == "W"


def test_clean_question_set_has_no_issues(doc):
    questions = [
        Question(
            id="q1",
            query="How is overtime paid?",
            gold_snippets=["Overtime is paid at 1.5x the regular hourly rate"],
        ),
    ]
    report = validate_questions(questions, [doc])
    assert report.ok
    assert report.issues == []
    assert report.num_scored == 1 and report.num_gold_snippets == 1


def test_planted_errors_are_all_found(doc):
    questions = [
        # 1. drifted snippet: one word changed -> should be an error with a fix
        Question(
            id="q1",
            query="Overtime?",
            gold_snippets=["Overtime is paid at 2.5x the regular hourly rate"],
        ),
        # 2. duplicate id
        Question(
            id="q1",
            query="Notice?",
            gold_snippets=["written notice at least 30 days prior to termination"],
        ),
        # 3. snippet absent from the corpus entirely
        Question(
            id="q3",
            query="Dress code?",
            gold_snippets=["employees must wear a blue tie on Fridays without exception"],
        ),
    ]
    report = validate_questions(questions, [doc])
    kinds = {i.kind for i in report.issues}
    assert "duplicate_id" in kinds
    assert "not_found" in kinds
    assert not report.ok

    not_found_ids = {i.question_id for i in report.issues if i.kind == "not_found"}
    assert "q3" in not_found_ids


def test_drifted_snippet_suggests_verbatim_text(doc):
    questions = [
        Question(
            id="q1",
            query="Overtime?",
            gold_snippets=["Overtime is paid at 2.5x the regular hourley rate"],
        ),
    ]
    report = validate_questions(questions, [doc])
    issues = [i for i in report.issues if i.suggestion]
    assert issues, "expected a correction suggestion"
    suggestion = issues[0].suggestion
    assert suggestion in DOC_TEXT  # verbatim, ready to paste
    assert "1.5x" in suggestion
    assert issues[0].location.startswith("handbook:")


def test_missing_gold_and_short_gold_are_warnings(doc):
    questions = [
        Question(id="q1", query="No gold here"),
        Question(id="q2", query="Short", gold_snippets=["Handbook"]),
    ]
    report = validate_questions(questions, [doc])
    kinds = {i.kind for i in report.issues}
    assert "no_gold" in kinds
    assert "too_short" in kinds
    assert report.ok  # warnings only: exit code stays 0


def test_ambiguous_snippet_across_documents(doc):
    other = Document(id="other", source_path="other.md", text=DOC_TEXT)
    questions = [
        Question(
            id="q1",
            query="Overtime?",
            gold_snippets=["Overtime is paid at 1.5x the regular hourly rate"],
        ),
    ]
    report = validate_questions(questions, [doc, other])
    assert any(i.kind == "ambiguous" for i in report.issues)


def test_bundled_example_question_set_is_valid():
    """The shipped corpus must always pass its own validator."""
    documents = load_documents(EXAMPLES / "corpus")
    questions = load_questions(EXAMPLES / "questions.yaml")
    report = validate_questions(questions, documents)
    assert report.ok, [i.message for i in report.errors][:5]


def test_no_suggestion_when_nothing_plausible(doc):
    """A snippet from a different universe must not get a misleading 'fix'."""
    questions = [
        Question(
            id="q1",
            query="Dividends?",
            gold_snippets=["the quarterly dividend is paid in arrears to holders"],
        ),
    ]
    report = validate_questions(questions, [doc])
    issue = next(i for i in report.issues if i.kind == "not_found")
    assert issue.suggestion is None
    assert "no similar passage" in issue.message


def test_suggestion_never_starts_mid_word(doc):
    # Truncated words at both ends, plus a wrong number so it is not contained.
    questions = [
        Question(
            id="q1",
            query="Notice?",
            gold_snippets=["ritten notice at least 45 days prior to terminatio"],
        ),
    ]
    report = validate_questions(questions, [doc])
    suggestion = next(i.suggestion for i in report.issues if i.suggestion)
    assert suggestion in DOC_TEXT
    assert suggestion.startswith("written")  # snapped left to the word start
    assert suggestion.endswith("termination.")  # and right to the word end
