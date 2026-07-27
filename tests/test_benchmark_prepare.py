"""Conversion logic for the public benchmarks (roadmap phase 7).

These run offline on small inline fixtures. The conversion is where a benchmark
result can go quietly wrong: a gold span that is not verbatim, or a two-word span
that matches everything, produces numbers that look fine and mean nothing.
"""

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parent.parent / "scripts" / "benchmarks"
sys.path.insert(0, str(SCRIPTS))

prepare_qasper = pytest.importorskip("prepare_qasper")
prepare_cuad = pytest.importorskip("prepare_cuad")
from common import ConversionStats  # noqa: E402


def _paper(**overrides) -> dict:
    paper = {
        "title": "A Study of Chunking",
        "abstract": "We study how documents are split.",
        "full_text": [
            {
                "section_name": "Experimental Setup",
                "paragraphs": [
                    "We evaluate on three corpora of technical documentation.",
                    "Each corpus contains at least one thousand paragraphs of prose.",
                ],
            }
        ],
        "qas": [],
    }
    paper.update(overrides)
    return paper


def _qa(evidence: list[str], unanswerable: bool = False) -> dict:
    return {
        "question": "which corpora are used?",
        "answers": [{"answer": {"unanswerable": unanswerable, "evidence": evidence}}],
    }


# --- QASPER --------------------------------------------------------------------


def test_rendered_paper_contains_every_paragraph_verbatim():
    paper = _paper()
    text = prepare_qasper.render_paper(paper)
    for section in paper["full_text"]:
        for paragraph in section["paragraphs"]:
            assert paragraph in text


def test_evidence_is_kept_when_verbatim():
    evidence = "We evaluate on three corpora of technical documentation."
    paper = _paper(qas=[_qa([evidence])])
    stats = ConversionStats()

    _, questions = prepare_qasper.convert({"1234.5678": paper}, ["1234.5678"], stats)

    assert len(questions) == 1
    assert questions[0]["gold_snippets"] == [evidence]
    assert questions[0]["tags"] == ["src:1234.5678", "qasper"]


def test_evidence_that_is_not_verbatim_is_dropped_not_fuzzy_matched():
    paper = _paper(qas=[_qa(["We evaluate on three corpora of MEDICAL documentation."])])
    stats = ConversionStats()

    _, questions = prepare_qasper.convert({"1": paper}, ["1"], stats)

    assert questions == []
    assert any("verbatim" in reason for reason in stats.dropped)


def test_figure_caption_evidence_is_dropped():
    """FLOAT SELECTED spans are captions, absent from the rendered body text."""
    paper = _paper(qas=[_qa(["FLOAT SELECTED: Table 1: Corpus statistics."])])
    stats = ConversionStats()

    _, questions = prepare_qasper.convert({"1": paper}, ["1"], stats)

    assert questions == []


def test_unanswerable_question_is_dropped():
    paper = _paper(qas=[_qa([], unanswerable=True)])
    stats = ConversionStats()

    _, questions = prepare_qasper.convert({"1": paper}, ["1"], stats)

    assert questions == []


def test_short_evidence_is_dropped_by_the_token_floor():
    """Regression: two QASPER annotations cite only the heading 'Experimental Setup',
    which occurs in 6 of 30 sampled papers and would score as a hit everywhere."""
    paper = _paper(qas=[_qa(["Experimental Setup"])])

    stats = ConversionStats()
    _, filtered = prepare_qasper.convert({"1": paper}, ["1"], stats, min_gold_tokens=5)
    assert filtered == []

    stats_raw = ConversionStats()
    _, unfiltered = prepare_qasper.convert({"1": paper}, ["1"], stats_raw, min_gold_tokens=0)
    assert len(unfiltered) == 1, "--min-gold-tokens 0 must reproduce the unfiltered set"


def test_first_annotator_with_evidence_wins_over_the_union():
    """Unioning annotators would inflate gold_total with their disagreement."""
    first = "We evaluate on three corpora of technical documentation."
    second = "Each corpus contains at least one thousand paragraphs of prose."
    paper = _paper(
        qas=[
            {
                "question": "which corpora?",
                "answers": [
                    {"answer": {"unanswerable": True, "evidence": []}},
                    {"answer": {"unanswerable": False, "evidence": [first]}},
                    {"answer": {"unanswerable": False, "evidence": [second]}},
                ],
            }
        ]
    )
    stats = ConversionStats()

    _, questions = prepare_qasper.convert({"1": paper}, ["1"], stats)

    assert questions[0]["gold_snippets"] == [first]


# --- CUAD ----------------------------------------------------------------------


def _contract(qas: list[dict], context: str) -> dict:
    return {"title": "ACME_Agreement", "paragraphs": [{"context": context, "qas": qas}]}


CONTEXT = (
    "This Agreement is governed by the laws of the State of New York. "
    "Either party may terminate upon ninety (90) days written notice to the other party."
)


def test_cuad_span_is_taken_from_the_annotated_offset():
    span = "governed by the laws of the State of New York"
    start = CONTEXT.index(span)
    contract = _contract(
        [{"question": "Governing Law?", "answers": [{"text": span, "answer_start": start}]}],
        CONTEXT,
    )
    stats = ConversionStats()

    context, questions = prepare_cuad.convert_contract(contract, stats, min_gold_tokens=5)

    assert context == CONTEXT
    assert questions[0]["gold_snippets"] == [span]


def test_cuad_impossible_category_is_dropped():
    contract = _contract(
        [{"question": "Governing Law?", "answers": [], "is_impossible": True}], CONTEXT
    )
    stats = ConversionStats()

    assert prepare_cuad.convert_contract(contract, stats, min_gold_tokens=5) is None


def test_cuad_duplicate_spans_are_deduplicated():
    span = "governed by the laws of the State of New York"
    start = CONTEXT.index(span)
    contract = _contract(
        [
            {
                "question": "Governing Law?",
                "answers": [
                    {"text": span, "answer_start": start},
                    {"text": span, "answer_start": start},
                ],
            }
        ],
        CONTEXT,
    )
    stats = ConversionStats()

    _, questions = prepare_cuad.convert_contract(contract, stats, min_gold_tokens=5)

    assert questions[0]["gold_snippets"] == [span]


def test_cuad_short_span_is_dropped():
    contract = _contract(
        [{"question": "Document Name?", "answers": [{"text": "This", "answer_start": 0}]}],
        CONTEXT,
    )
    stats = ConversionStats()

    assert prepare_cuad.convert_contract(contract, stats, min_gold_tokens=5) is None


def test_conversion_stats_report_the_drop_rate():
    stats = ConversionStats(documents=2, questions_seen=10, questions_kept=7)
    stats.drop("because", 3)

    rendered = stats.render()

    assert "70.0%" in rendered
    assert "3  because" in rendered
