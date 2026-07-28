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


def test_data_rate_is_an_amount_not_a_count():
    """Regression: the unit alternation had no right boundary, so "5 Gbps" matched
    as "5 Gb" and the draft came out as "How many is throughput capped at?"."""
    q = build_query("Per-flow throughput is capped at 5 Gbps everywhere.")
    assert q is not None and q.startswith("How much"), q


def test_data_volume_is_an_amount_not_a_count():
    q = build_query("Environment variables are capped at 4 KB total.")
    assert q is not None and q.startswith("How much"), q


def test_countable_unit_stays_how_many():
    q = build_query("Each project is limited to 50 documents per import.")
    assert q == "How many is each project limited to?", q


def test_compound_auxiliary_inverts_around_the_subject():
    """ "can be imported" must become "can X be imported", not "can be X imported"."""
    q = build_query("Custom images can be imported up to 500 GB.")
    assert q is not None
    assert "can custom images be imported" in q, q


def test_keeps_the_article_of_the_subject():
    q = build_query("A report of a production outage is acknowledged within one (1) hour.")
    assert q is not None and q.startswith("How long is a report"), q


def test_acronym_subject_keeps_its_case():
    q = build_query("SLA credits are capped at thirty percent of the monthly fee.")
    assert q is not None and "SLA credits" in q, q


def test_rejects_focus_inside_a_subordinate_clause():
    """The fact belongs to the inner clause, so the question would misattribute it."""
    assert (
        build_query(
            "Class imbalance is handled automatically when the minority class "
            "falls below ten percent."
        )
        is None
    )


def test_rejects_dangling_comparative():
    assert (
        build_query(
            "The revision may not exceed the greater of five percent (5%) and the index change."
        )
        is None
    )


def test_predicate_stops_at_a_parenthetical_aside():
    q = build_query("The credits are capped, in each calendar month, at thirty percent (30%).")
    assert q is not None
    assert "calendar month" not in q, q


def test_amount_strands_its_preposition():
    q = build_query("Mileage is reimbursed at 0.38 euros per kilometer.")
    assert q is not None and q.endswith("reimbursed at?"), q


def test_duration_does_not_strand_its_preposition():
    """ "payable within?" reads worse than "payable?" - only "at" strands well."""
    q = build_query("Each invoice is payable within thirty (30) days of the invoice date.")
    assert q is not None and q.endswith("payable?"), q


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


def test_repeated_boilerplate_yields_one_question():
    """A 10-K repeats "The information required by this Item will be included..."
    under several headings; the template turned each into the same query."""
    from chunklab.eval.qa_gen import generate_questions
    from chunklab.models import Document

    boilerplate = (
        "The information required by this Item will be included in the proxy "
        "statement within 120 days after the fiscal year end.\n\n"
    )
    filler = "Unrelated narrative text about operations and segments.\n\n" * 12
    doc = Document(
        id="filing",
        source_path="filing.md",
        text=(boilerplate + filler) * 4,
        elements=[],
        metadata={},
    )

    questions = generate_questions([doc], n=10)

    queries = [q.query for q in questions]
    assert len(queries) == len(set(queries)), f"duplicate queries generated: {queries}"
