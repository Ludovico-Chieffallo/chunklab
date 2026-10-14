"""A chunk shorter than the gold snippet cannot contain it.

rapidfuzz's `partial_ratio` looks for the shorter string inside the longer one. While
chunks are longer than snippets that is containment, but a chunk shorter than the
snippet turned the roles around: the chunk was looked for inside the snippet, and any
fragment of the answer scored 100. On the bundled example at `chunk_size: 24`, 15 of
63 hits were such fragments - "are retried twice with exponential backoff", 42
characters, was counted as the whole 216-character answer. That inflates the small
sizes a chunk-size sweep is trying to judge, and it hid every answer split across
two small chunks, because each half already counted as found.
"""

from rapidfuzz import fuzz

from chunklab.eval.gold_match import normalize, score_question, snippet_in_text
from chunklab.models import Chunk, Document, Question, RetrievedChunk
from chunklab.validation import validate_questions

ANSWER = (
    "asynchronous invocations are retried twice with exponential backoff. After the final "
    "failure the event goes to the project's dead-letter queue if one is configured"
)
FIRST, SECOND = ANSWER.split(". ", 1)


def _retrieved(*texts: str) -> list[RetrievedChunk]:
    retrieved, at = [], 0
    for rank, text in enumerate(texts, start=1):
        chunk = Chunk(
            id=f"d:s:{rank}",
            doc_id="d",
            text=text,
            token_count=len(text.split()),
            char_span=(at, at + len(text)),
            strategy="s",
        )
        retrieved.append(RetrievedChunk(chunk=chunk, score=1.0 / rank, rank=rank))
        at += len(text) + 1
    return retrieved


def test_a_fragment_of_the_snippet_is_not_the_snippet():
    fragment = "are retried twice with exponential backoff"
    assert fuzz.partial_ratio(normalize(ANSWER), normalize(fragment)) == 100  # the old rule

    assert not snippet_in_text(ANSWER, fragment)


def test_a_chunk_that_is_the_snippet_with_a_typo_still_matches():
    """Shorter than the snippet by one letter, and still the whole answer."""
    assert snippet_in_text(
        "written notice at least 30 days prior to termination",
        "writen notice at least 30 days prior to termination",
    )


def test_an_answer_split_across_two_small_chunks_is_reported_as_split():
    question = Question(id="q", query="What happens to failed async calls?", gold_snippets=[ANSWER])

    result = score_question(question, _retrieved(FIRST + ".", SECOND), "s")

    assert not result.hit and result.gold_found_count == 0
    assert result.split_across_chunks


def test_validate_and_scoring_agree_on_a_document_shorter_than_the_snippet():
    document = Document(id="doc", source_path="doc.md", text=FIRST + ".")
    question = Question(id="q1", query="?", gold_snippets=[ANSWER])

    report = validate_questions([question], [document])

    assert not report.ok
    assert not snippet_in_text(ANSWER, document.text)
