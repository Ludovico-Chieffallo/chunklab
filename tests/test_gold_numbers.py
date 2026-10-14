"""A passage that states other numbers is not the gold snippet.

Fuzzy containment exists to forgive a typo, and at the default 0.90 threshold it
forgave a changed number too: "within 14 days" was found in "within 30 days", "10 MB"
in "50 MB", "1.5x" in "2.5x". On a 26-paper corpus, a snippet for the high-dose result
of a clinical trial (80%, 84% and 96% reductions) aligned at 91% with the sentence just
before it, which reports the low dose (47%, 52% and 56%), and was counted as found by
three cells of the matrix, the leading one among them. `validate` made it worse: it
called the match a harmless drift and offered the low-dose sentence as the verbatim
text to paste.

Exact containment had the same flaw at the edges of a snippet: "5 business days" is a
substring of "15 business days".
"""

import random

import pytest
from rapidfuzz import fuzz

from chunklab.eval.gold_match import normalize, score_question, snippet_in_text
from chunklab.models import Chunk, Document, Question, RetrievedChunk
from chunklab.validation import validate_questions

TRIAL = (
    "On day 28, 47%, 52%, and 56% reductions in blood TTR concentrations were seen "
    "at the low dose, and 80%, 84%, and 96% reductions were seen at the high dose."
)
HIGH_DOSE = "80%, 84%, and 96% reductions in blood TTR concentrations"

OTHER_NUMBERS = [
    (
        "Refunds are issued within 14 days of the request",
        "Refunds are issued within 30 days of the request.",
    ),
    (
        "Uploads are limited to 10 MB per file on every plan",
        "Uploads are limited to 50 MB per file on every plan.",
    ),
    (
        "Overtime is paid at 1.5x the regular hourly rate",
        "Overtime is paid at 2.5x the regular hourly rate.",
    ),
    (
        "The 2023 agreement renews for 30 days unless cancelled",
        "The 2024 agreement renews for 90 days unless cancelled.",
    ),
    (HIGH_DOSE, TRIAL),
]

#: Substrings, so plain containment accepted them, but each one changes a number.
CUT_NUMBERS = [
    (
        "5 business days after the invoice date",
        "Payment is due 15 business days after the invoice date.",
    ),
    (
        "5x the regular hourly rate for night shifts",
        "Paid at 1.5x the regular hourly rate for night shifts.",
    ),
    (
        "the warranty covers parts and labour for 2",
        "The warranty covers parts and labour for 24 months.",
    ),
]

SAME_NUMBERS = [
    # A typo in the words, the numbers as written.
    (
        "Refunds are isued within 14 days of the request",
        "Refunds are issued within 14 days of the request.",
    ),
    ("80%, 84%, and 96% reductions were seen at the hihg dose", TRIAL),
    # Other numbers around the snippet do not matter.
    (
        "issued within 14 days of the request",
        "Refunds (policy 7) are issued within 14 days of the request, 30 in the EU.",
    ),
    (
        "covered for 24 months from delivery",
        "Parts are covered for 24 months from delivery (2 for batteries).",
    ),
]


# --- scoring --------------------------------------------------------------------------


@pytest.mark.parametrize(("snippet", "passage"), OTHER_NUMBERS)
def test_a_passage_with_other_numbers_is_not_the_snippet(snippet, passage):
    # The old rule accepted every one of these: the words are close enough.
    assert fuzz.partial_ratio(normalize(snippet), normalize(passage)) >= 90

    assert not snippet_in_text(snippet, passage, fuzzy_threshold=0.90)


@pytest.mark.parametrize(("snippet", "passage"), CUT_NUMBERS)
def test_a_substring_that_cuts_a_number_is_not_the_snippet(snippet, passage):
    assert normalize(snippet) in normalize(passage)

    assert not snippet_in_text(snippet, passage)


@pytest.mark.parametrize(("snippet", "passage"), SAME_NUMBERS)
def test_the_same_numbers_with_drift_around_them_still_match(snippet, passage):
    assert snippet_in_text(snippet, passage)


def test_the_right_passage_is_found_behind_a_closer_one_with_other_numbers():
    """Templated text: the closest alignment states other numbers, the right passage
    is further down and only fuzzily equal. The second is the answer."""
    snippet = "Uploads are limited to 10 MB per file on every plan"
    chunk = (
        "Uploads are limited to 50 MB per file on every plan. "
        "Legacy accounts: uploads are limted to 10 MB per fle on every plan."
    )

    assert snippet_in_text(snippet, chunk)


def test_the_trial_snippet_is_not_found_in_the_low_dose_sentence():
    low_dose = TRIAL.split(", and 80%")[0] + "."
    chunk = Chunk(
        id="d:s:0",
        doc_id="d",
        text=low_dose,
        token_count=len(low_dose.split()),
        char_span=(0, len(low_dose)),
        strategy="s",
    )
    question = Question(id="q", query="What did the high dose achieve?", gold_snippets=[HIGH_DOSE])

    result = score_question(question, [RetrievedChunk(chunk=chunk, score=1.0, rank=1)], "s")

    assert not result.hit and result.gold_found_count == 0


def test_snippets_without_numbers_score_exactly_as_before():
    """The rule only ever says no to a number: elsewhere nothing moves."""
    rng = random.Random(0)
    words = "the a of days within refunds rate and in blood notice written prior to".split()
    for _ in range(3000):
        snippet = " ".join(rng.choices(words, k=rng.randint(1, 10)))
        text = " ".join(rng.choices(words, k=rng.randint(1, 40)))
        g, c = normalize(snippet), normalize(text)
        if len(c) < len(g):
            continue  # a text too short to hold the snippet: see test_gold_fragments.py
        before = g in c or fuzz.partial_ratio(g, c) / 100.0 >= 0.90

        assert snippet_in_text(snippet, text) == before, (snippet, text)


# --- validate -------------------------------------------------------------------------


def _validate(snippet: str, text: str):
    document = Document(id="doc", source_path="doc.md", text=text)
    question = Question(id="q1", query="?", gold_snippets=[snippet])
    return validate_questions([question], [document])


def test_validate_calls_a_match_with_other_numbers_an_error():
    report = _validate(HIGH_DOSE, TRIAL)

    (issue,) = report.issues
    assert issue.kind == "numbers_differ" and issue.severity == "error"
    assert "80, 84, 96" in issue.message and "47, 52, 56" in issue.message
    assert not report.ok


@pytest.mark.parametrize(("snippet", "passage"), CUT_NUMBERS)
def test_validate_calls_a_substring_that_cuts_a_number_an_error(snippet, passage):
    report = _validate(snippet, passage)

    assert [i.kind for i in report.errors] == ["numbers_differ"]


def test_validate_points_at_the_passage_with_the_snippets_numbers():
    snippet = "Uploads are limited to 10 MB per file on every plan"
    text = (
        "Uploads are limited to 50 MB per file on every plan. "
        "Legacy accounts: uploads are limted to 10 MB per fle on every plan."
    )

    (issue,) = _validate(snippet, text).issues

    assert issue.kind == "fuzzy_only"
    assert "10 MB" in issue.suggestion and "50 MB" not in issue.suggestion


@pytest.mark.parametrize(("snippet", "passage"), OTHER_NUMBERS + CUT_NUMBERS + SAME_NUMBERS)
def test_validate_and_scoring_agree(snippet, passage):
    """`validate` passing a snippet must mean scoring can find it, and the reverse."""
    report = _validate(snippet, passage)

    assert report.ok == snippet_in_text(snippet, passage)
