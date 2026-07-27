"""Tests for the heuristic question bootstrapper (roadmap 3.2).

The generator's contract is narrow but strict: gold snippets must be verbatim
source text, queries must be typed factual questions, and everything must be
marked unreviewed. Question *quality* is a human judgement (checkpoint CP3);
these tests pin the properties that can be checked mechanically.
"""

from pathlib import Path

import pytest
import yaml

from chunklab.eval.gold_match import normalize
from chunklab.eval.qa_gen import build_query, dump_questions_yaml, generate_questions
from chunklab.loaders.registry import load_documents
from chunklab.models import Document

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture(scope="module")
def corpus() -> list[Document]:
    return load_documents(EXAMPLES / "corpus")


@pytest.fixture(scope="module")
def drafted(corpus):
    return generate_questions(corpus, n=20)


def test_gold_snippets_are_verbatim(drafted, corpus):
    joined = " \n ".join(normalize(d.text) for d in corpus)
    for q in drafted:
        for gold in q.gold_snippets:
            assert normalize(gold) in joined, f"{q.id}: gold snippet is not verbatim"


def test_questions_are_marked_unreviewed(drafted):
    assert drafted
    assert all(q.reviewed is False for q in drafted)
    assert all("generated" in q.tags for q in drafted)


def test_ids_are_unique_and_sources_tagged(drafted):
    ids = [q.id for q in drafted]
    assert len(ids) == len(set(ids))
    assert all(any(t.startswith("src:") for t in q.tags) for q in drafted)


def test_questions_spread_across_documents(drafted):
    sources = {t for q in drafted for t in q.tags if t.startswith("src:")}
    assert len(sources) >= 3, f"drafts clustered in too few documents: {sources}"


def test_every_query_is_a_typed_question(drafted):
    for q in drafted:
        assert q.query.endswith("?")
        assert q.query.split()[0] in {"How", "What", "When"}
        assert len(q.query.split()) <= 16


def test_drafted_set_passes_validation(drafted, corpus):
    """A draft must be immediately usable: no errors from `chunklab validate`."""
    from chunklab.validation import validate_questions

    report = validate_questions(drafted, corpus)
    assert report.ok, [i.message for i in report.errors][:3]


# --- build_query unit behavior -------------------------------------------------


def test_duration_becomes_how_long():
    q = build_query("Customer records are retained for seven years after the relationship ends.")
    assert q == "How long are customer records retained?"


def test_money_becomes_how_much():
    q = build_query("Mileage is reimbursed at 0.38 euros per kilometer.")
    assert q is not None and q.startswith("How much")


def test_percentage_focus():
    q = build_query("Encryption at rest is effectively universal at ninety-six percent adoption.")
    assert q is not None and q.startswith("What percentage")


def test_spelled_number_with_parenthetical_numeral():
    """Contracts write 'thirty (30) days' - the unit must still be seen."""
    q = build_query("Each invoice is payable within thirty (30) days of the invoice date.")
    assert q is not None and q.startswith("How long")


def test_rejects_bare_number_without_unit():
    assert build_query("Timestamps are RFC 3339 in UTC.") is None


def test_rejects_identifier_numbers():
    assert build_query("Objects are encrypted at rest with AES-256 using platform keys.") is None


def test_rejects_subordinate_opener():
    assert build_query("With schema_mode set to infer, types are inferred from 100 rows.") is None


def test_rejects_markdown_question_heading():
    assert build_query("**How long are logs kept?** Logs are kept for thirty days.") is None


def test_yaml_dump_roundtrips(drafted):
    text = dump_questions_yaml(drafted)
    parsed = yaml.safe_load(text)
    assert len(parsed["questions"]) == len(drafted)
    assert all(q["reviewed"] is False for q in parsed["questions"])
    assert parsed["questions"][0]["gold_snippets"] == drafted[0].gold_snippets


def test_runner_warns_about_unreviewed_questions(handbook, drafted):
    from chunklab.config import default_config
    from chunklab.models import Question
    from chunklab.runner import run_evaluation

    config = default_config()
    config.embedding.backend = "fake"
    questions = [
        Question(
            id="q1",
            query="How is overtime compensated?",
            gold_snippets=["Overtime is paid at 1.5x the regular hourly rate"],
            reviewed=False,
        )
    ]
    report = run_evaluation([handbook], questions, config)
    assert any("reviewed: false" in w for w in report.warnings)
