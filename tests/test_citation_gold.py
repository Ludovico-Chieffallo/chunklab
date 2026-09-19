"""A gold snippet that only lives in a reference list is not evidence.

Found on a run over 26 CRISPR review papers. Two of the twenty scored
questions had a *cited paper's title* as their gold snippet — 'Double nicking
by RNA-guided CRISPR Cas9…' (Ran et al. 2013) and 'Functional repair of CFTR
by CRISPR/Cas9…' (Schwank et al. 2013) — each appearing in the bibliographies
of seven to nine of the papers and nowhere in any body text.

The two failed in opposite directions and both corrupted the measurement. The
first was found by *every* cell of the 15-cell matrix, because any strategy
retrieves some bibliography chunk, so it added a constant to every score. The
second was found by *none*, because a list of citations sits nowhere near the
question, so it subtracted one. `validate` had flagged both only as
`ambiguous` ("appears in 9 documents"), which was too weak to stop the run.

Measured before the rule was written: 2 of 2 known artifacts caught, 0 false
positives over the 152 gold snippets of the example corpus and the test
handbook.
"""

import pytest

from chunklab.loaders.text import TextLoader
from chunklab.models import Question
from chunklab.validation import in_bibliography, validate_questions

PAPER = """# Genome editing review

## Results

Cas9 nickase with two sgRNAs lowers off-target cleavage by several orders of
magnitude, because a single nick is repaired without a double-strand break.

## References

- Ran FA, Hsu PD, Lin CY, et al. Double nicking by RNA-guided CRISPR Cas9 for
  enhanced genome editing specificity. Cell 2013;154:1380-1389.
- Schwank G, Koo BK, Sasselli V. Functional repair of CFTR in organoids.
  Cell Stem Cell 2013;13:653-658.
"""

CITED_TITLE = "Double nicking by RNA-guided CRISPR Cas9 for enhanced genome editing specificity"
BODY_SENTENCE = "a single nick is repaired without a double-strand break"


def _doc(text: str, tmp_path, name="paper.md"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return TextLoader().load(path)


def _validate(text, snippet, tmp_path, **kw):
    doc = _doc(text, tmp_path)
    question = Question(id="q1", query="how does nickase help?", gold_snippets=[snippet])
    return validate_questions([question], [doc], **kw)


def test_a_cited_title_is_an_error(tmp_path):
    report = _validate(PAPER, CITED_TITLE, tmp_path)

    assert not report.ok, "a snippet that cannot be answered must gate CI"
    issue = report.errors[0]
    assert issue.kind == "citation_only"
    assert "reference lists" in issue.message
    assert issue.location and issue.location.startswith("paper:")


def test_a_body_sentence_is_left_alone(tmp_path):
    report = _validate(PAPER, BODY_SENTENCE, tmp_path)

    assert report.ok
    assert not [i for i in report.issues if i.kind == "citation_only"]


def test_a_snippet_also_present_in_the_body_is_legitimate(tmp_path):
    """Only snippets whose *every* occurrence is a citation are refused; a
    phrase the paper both states and cites is ordinary evidence."""
    text = PAPER.replace(
        "## Results\n",
        f"## Results\n\nWe confirm {CITED_TITLE} as reported previously.\n",
    )
    report = _validate(text, CITED_TITLE, tmp_path)

    assert not [i for i in report.issues if i.kind == "citation_only"]


@pytest.mark.parametrize(
    "heading",
    ["References", "REFERENCES", "Bibliography", "5. References", "Literature Cited"],
)
def test_reference_heading_spellings(heading, tmp_path):
    report = _validate(PAPER.replace("## References", f"## {heading}"), CITED_TITLE, tmp_path)

    assert [i for i in report.issues if i.kind == "citation_only"], heading


def test_a_running_page_header_does_not_mask_the_reference_list(tmp_path):
    """The regression that set the rule. PDF converters promote running page
    headers to headings; one landing inside the bibliography made the innermost
    heading the journal name, so only the full trail can be trusted."""
    text = PAPER.replace(
        "- Schwank G",
        "### BioMed Research International\n\n- Schwank G",
    )
    doc = _doc(text, tmp_path)
    offset = doc.text.index("Double nicking")

    assert in_bibliography(doc, offset)
    report = _validate(text, CITED_TITLE, tmp_path)
    assert [i for i in report.issues if i.kind == "citation_only"]


def test_a_document_without_headings_is_never_flagged(tmp_path):
    """With no structure there is no evidence of a bibliography, and guessing
    would turn an unreadable PDF into a wall of false errors."""
    flat = PAPER.replace("#", "")
    report = _validate(flat, CITED_TITLE, tmp_path)

    assert not [i for i in report.issues if i.kind == "citation_only"]
