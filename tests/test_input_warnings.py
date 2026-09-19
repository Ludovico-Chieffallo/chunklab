"""What `chunklab run` must tell you about the documents you gave it.

Everything here was found by running the tool over 26 CRISPR review papers —
real, messy, converted-from-PDF input rather than the handwritten corpus in
`examples/`. Each case is something the run silently accepted.
"""

import pytest

from chunklab.config import default_config
from chunklab.models import Document, Question
from chunklab.runner import run_evaluation

CLEAN = "Overtime is paid at 1.5x the regular hourly rate for all hours beyond forty."


def _doc(doc_id: str, text: str, **metadata) -> Document:
    return Document(id=doc_id, source_path=f"/tmp/{doc_id}", text=text, metadata=metadata)


def _questions(n: int = 2) -> list[Question]:
    return [
        Question(
            id=f"q{i}",
            query="how is overtime paid?",
            gold_snippets=["paid at 1.5x the regular hourly rate"],
        )
        for i in range(n)
    ]


def _run(documents: list[Document]):
    config = default_config()
    config.embedding.backend = "fake"
    return run_evaluation(documents, _questions(), config)


def _warning(report, needle: str) -> str | None:
    return next((w for w in report.warnings if needle in w), None)


# --- undecodable text (U+FFFD) --------------------------------------------------


def test_replacement_characters_are_reported():
    """The existing encoding warning reads a flag only the text loader sets, so
    13 of 26 damaged PDFs went through without a word. On that corpus U+FFFD
    replaced primes, en dashes, '≈' and a whole line of Chinese."""
    report = _run([_doc("paper", CLEAN + " identity was �20% across �Cas9 variants.")])

    warning = _warning(report, "U+FFFD")
    assert warning is not None
    assert "paper" in warning and "2" in warning


def test_clean_documents_raise_no_undecodable_warning():
    report = _run([_doc("clean", CLEAN)])

    assert _warning(report, "U+FFFD") is None


def test_the_worst_document_is_named_first():
    """Only the first few ids fit in the warning, so they must be the ones worth
    opening — ranked by density, not by how many characters the file happens to
    have."""
    light = _doc("light", CLEAN + " a�b " + "filler text here. " * 400)
    heavy = _doc("heavy", CLEAN + " ���")

    report = _run([light, heavy])
    warning = _warning(report, "U+FFFD")

    assert warning is not None
    assert warning.index("heavy") < warning.index("light")


def test_the_total_counts_every_occurrence():
    report = _run([_doc("a", CLEAN + " ��"), _doc("b", CLEAN + " �")])

    warning = _warning(report, "U+FFFD")
    assert "2 document(s) hold 3 replacement character(s)" in warning


def test_the_text_loader_fallback_warning_still_fires_separately():
    """The two warnings answer different questions — which file was decoded by
    guesswork, and which text came out damaged — so neither replaces the other."""
    report = _run([_doc("guessed", CLEAN + " caf�", encoding="cp1252")])

    assert _warning(report, "decoded by fallback") is not None
    assert _warning(report, "U+FFFD") is not None


@pytest.mark.parametrize("text", ["�" + CLEAN, CLEAN + "�"])
def test_replacement_at_either_edge_is_counted(text):
    assert _warning(_run([_doc("edge", text)]), "U+FFFD") is not None
