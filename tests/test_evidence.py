"""`find-evidence` locates candidate gold passages without using embeddings.

The gap it fills was measured: of 60 hand-written questions over 26 CRISPR
review papers, 40 had no gold snippet and were dropped from scoring, because
annotating them by hand was too slow to finish. Two thirds of a question set
lost to friction.

The no-embeddings rule is the part that matters. If the retriever under
evaluation chooses the passages it will later be scored against, that cell
reaches recall 1.0 by construction and the comparison between strategies
measures nothing. Candidates are found by searching the raw text for the
question's distinctive terms — how a human annotator works — and a person
still chooses.
"""

import pytest

from chunklab.evidence import (
    anchors_for,
    document_frequency,
    find_evidence,
)
from chunklab.loaders.text import TextLoader
from chunklab.models import Document, Question

PAPER = """# Editing review

## Results

Knockout of OsNramp5, which mediates the root uptake of cadmium, produced rice
lines with low grain accumulation. Yield was unaffected in the field trials.

The buffering capacity of 2-methylimidazole linkers drives endosomal escape.

## References

- Tang L, Mao B. Knockout of OsNramp5 using CRISPR/Cas9 produces low
  Cd-accumulating indica rice. Sci Rep 2017;7:14438.
"""


def _doc(text: str, tmp_path, name="paper.md") -> Document:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return TextLoader().load(path)


def _q(qid: str, query: str, gold=()) -> Question:
    return Question(id=qid, query=query, gold_snippets=list(gold))


# --- document frequency ---------------------------------------------------------


def test_frequency_counts_documents_not_occurrences():
    docs = [
        Document(id="a", source_path="a", text="cas9 cas9 cas9"),
        Document(id="b", source_path="b", text="cas9 and pam"),
    ]
    frequency = document_frequency(docs)

    assert frequency["cas9"] == 2
    assert frequency["pam"] == 1


# --- choosing anchors -----------------------------------------------------------


def test_terms_in_too_much_of_the_corpus_are_not_anchors():
    """On near-duplicate papers a term everyone uses finds noise, which is how
    'limitation' and 'traditional' buried the gene name that would have worked."""
    frequency = {"crispr": 20, "osnramp5": 1}
    anchors, _ = anchors_for("Why is CRISPR used for OsNramp5?", frequency, 20)

    assert [term for term, _ in anchors] == ["OsNramp5"]


def test_absent_terms_are_reported_separately():
    anchors, absent = anchors_for("What about exons 45-55?", {"exons": 1}, 8)

    assert [term for term, _ in anchors] == ["exons"]
    assert "45-55" in absent


def test_identifier_shaped_terms_win_ties():
    """At equal rarity, `GmFAD2-1A` is worth more than `rationale`."""
    frequency = {"rationale": 2, "gmfad2-1a": 2}
    anchors, _ = anchors_for("What is the rationale for GmFAD2-1A?", frequency, 20)

    assert [term for term, _ in anchors][0] == "GmFAD2-1A"


def test_stopwords_and_short_words_are_dropped():
    anchors, absent = anchors_for("How does it do so?", {"how": 1, "does": 1}, 8)

    assert anchors == []
    assert absent == []


# --- finding candidates ---------------------------------------------------------


def test_candidates_come_back_for_an_unannotated_question(tmp_path):
    doc = _doc(PAPER, tmp_path)
    [evidence] = find_evidence([_q("q1", "What does OsNramp5 knockout do to cadmium?")], [doc])

    assert evidence.anchored
    assert any("root uptake of cadmium" in c.text for c in evidence.candidates)


def test_questions_that_already_have_gold_are_skipped(tmp_path):
    doc = _doc(PAPER, tmp_path)
    questions = [_q("q1", "OsNramp5?", gold=["Yield was unaffected"]), _q("q2", "OsNramp5?")]

    assert [e.question_id for e in find_evidence(questions, [doc])] == ["q2"]


def test_reference_lists_are_never_offered(tmp_path):
    """A cited work's title is not evidence — `validate` refuses it outright, so
    offering it here would only waste the annotator's time."""
    doc = _doc(PAPER, tmp_path)
    [evidence] = find_evidence([_q("q1", "What about OsNramp5?")], [doc])

    assert not any("Sci Rep" in c.text for c in evidence.candidates)
    assert not any("14438" in c.text for c in evidence.candidates)


def test_passages_with_undecodable_characters_are_skipped(tmp_path):
    """A snippet copied out of damaged text cannot match exactly, so it is not a
    candidate however relevant it looks."""
    doc = _doc("# T\n\n## R\n\nThe ZIF-8 carrier gives �20% escape in cells.\n", tmp_path)
    [evidence] = find_evidence([_q("q1", "How does ZIF-8 help escape?")], [doc])

    assert evidence.anchored
    assert evidence.candidates == []


def test_table_rows_are_skipped(tmp_path):
    doc = _doc("# T\n\n## R\n\n|OsNramp5|yes|1|2|3|4|no|\n", tmp_path)
    [evidence] = find_evidence([_q("q1", "What about OsNramp5?")], [doc])

    assert evidence.candidates == []


def test_a_question_with_no_anchor_says_so(tmp_path):
    doc = _doc(PAPER, tmp_path)
    [evidence] = find_evidence([_q("q1", "What about exon 45-55 skipping in dystrophin?")], [doc])

    assert not evidence.anchored
    assert "dystrophin" in evidence.absent


@pytest.mark.parametrize("context", [0, 2])
def test_context_widens_the_passage(tmp_path, context):
    doc = _doc(PAPER, tmp_path)
    [evidence] = find_evidence(
        [_q("q1", "What about OsNramp5?")], [doc], context=context, max_candidates=1
    )
    text = evidence.candidates[0].text

    assert ("Yield was unaffected" in text) is (context > 0)


def test_candidate_count_is_capped(tmp_path):
    doc = _doc(PAPER + "\n\nOsNramp5 again. " * 20, tmp_path)
    [evidence] = find_evidence([_q("q1", "What about OsNramp5?")], [doc], max_candidates=2)

    assert len(evidence.candidates) <= 2


def test_candidates_carry_a_locatable_offset(tmp_path):
    doc = _doc(PAPER, tmp_path)
    [evidence] = find_evidence([_q("q1", "What about OsNramp5?")], [doc])
    candidate = evidence.candidates[0]

    assert doc.text[candidate.offset :].startswith(candidate.text.split(" ")[0])
