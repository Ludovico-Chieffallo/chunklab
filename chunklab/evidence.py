"""Locate candidate gold passages for questions you have already written.

`bootstrap` drafts questions from the documents and `validate` repairs a
snippet you already have. Between them sits the step that actually costs a
user their afternoon: *I wrote the questions, now find me the evidence.* On a
26-paper corpus, 40 of 60 hand-written questions had no gold snippet and were
silently dropped from scoring — the whole question set reduced to a third of
itself because annotating it by hand was too slow.

**This deliberately uses no embeddings.** If the retriever under evaluation
picks the passages it will later be scored against, the benchmark measures
itself: that cell goes to recall 1.0 by construction and the comparison
between strategies becomes meaningless. Candidates are found by searching the
raw text for the *distinctive* terms of the question, which is how a human
annotator works and what the QASPER protocol does. Nothing here decides
anything — a person still reads the candidates and chooses.
"""

import re
from dataclasses import dataclass, field

from chunklab.models import Document, Question
from chunklab.text_utils import sentence_spans
from chunklab.validation import in_bibliography

#: A term in more than this share of the corpus anchors nothing. On 26 review
#: papers about one subject, "limitation" and "traditional" match everywhere
#: and bury the gene name that would have found the passage.
ANCHOR_DOCUMENT_SHARE = 0.25

#: Function and question words carry no corpus signal at any frequency, so they
#: are dropped before the frequency test rather than after it. Everything
#: topical is left to `ANCHOR_DOCUMENT_SHARE`, which adapts to the corpus.
STOPWORDS = frozenset(
    """a an and are as at be been being but by can could did do does for from had has have
    how in into is it its of on or should such than that the their them then there these this
    those to was were what when where which who whom whose why will with would you your
    about after again against all also any because before between both during each few further
    here more most no nor not now only other out over own same so some too under until up very
    does doing done use used using""".split()
)

#: Terms shaped like an identifier — a digit, a hyphen, or an interior capital.
#: Gene names, compounds, assay names and clause numbers all look like this,
#: and they are what makes a question findable in prose that repeats itself.
_DISTINCTIVE = re.compile(r"\d|-|(?<=.)[A-Z]")

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-/']*")

#: Markdown table rows survive PDF conversion as pipe-laden lines. They are
#: never a readable gold snippet.
_TABLE_PIPES = 4


@dataclass(frozen=True)
class Candidate:
    """One passage a person might turn into a gold snippet."""

    term: str  # the anchor that found it
    doc_id: str
    offset: int  # char offset of the passage in the document
    text: str


@dataclass
class QuestionEvidence:
    question_id: str
    query: str
    anchors: list[tuple[str, int]] = field(default_factory=list)  # (term, documents)
    absent: list[str] = field(default_factory=list)  # query terms not in the corpus
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def anchored(self) -> bool:
        return bool(self.anchors)


def document_frequency(documents: list[Document]) -> dict[str, int]:
    """How many documents each lowercase term appears in."""
    counts: dict[str, int] = {}
    for document in documents:
        for term in {w.lower() for w in _WORD.findall(document.text)}:
            counts[term] = counts.get(term, 0) + 1
    return counts


def anchors_for(
    query: str, frequency: dict[str, int], num_documents: int, limit: int = 4
) -> tuple[list[tuple[str, int]], list[str]]:
    """The query's distinctive terms, rarest first, and the ones not in the corpus.

    Ranked by document frequency, with identifier-shaped terms preferred at
    equal frequency: on a corpus of near-duplicate papers, `OsNramp5` finds the
    passage and `strategy` finds noise.
    """
    ceiling = max(2, int(num_documents * ANCHOR_DOCUMENT_SHARE))
    seen: set[str] = set()
    scored: list[tuple[int, int, str]] = []
    absent: list[str] = []
    for word in _WORD.findall(query):
        lowered = word.lower()
        if lowered in STOPWORDS or len(word) < 3 or lowered in seen:
            continue
        seen.add(lowered)
        found = frequency.get(lowered, 0)
        if found == 0:
            absent.append(word)
        elif found <= ceiling:
            scored.append((found, 0 if _DISTINCTIVE.search(word) else 1, word))
    scored.sort()
    return [(word, found) for found, _, word in scored[:limit]], absent


def _passages(document: Document, term: str, context: int, per_term: int) -> list[Candidate]:
    """Sentences containing `term`, with `context` neighbours either side.

    Reference lists and passages holding U+FFFD are skipped: the first is a
    cited work's title rather than evidence, and the second cannot be copied
    into a gold snippet that will match exactly.
    """
    spans = sentence_spans(document.text)
    sentences = [document.text[start:end] for start, end in spans]
    needle = term.lower()
    found: list[Candidate] = []
    for i, sentence in enumerate(sentences):
        if needle not in sentence.lower():
            continue
        if in_bibliography(document, spans[i][0]):
            continue
        low, high = max(0, i - context), min(len(sentences), i + context + 1)
        text = " ".join(sentences[low:high])
        if "�" in text or text.count("|") > _TABLE_PIPES:
            continue
        found.append(Candidate(term=term, doc_id=document.id, offset=spans[low][0], text=text))
        if len(found) >= per_term:
            break
    return found


def find_evidence(
    questions: list[Question],
    documents: list[Document],
    max_candidates: int = 5,
    context: int = 1,
    per_term: int = 3,
) -> list[QuestionEvidence]:
    """Candidate passages for every question that has no gold snippet yet."""
    frequency = document_frequency(documents)
    results: list[QuestionEvidence] = []
    for question in questions:
        if question.gold_snippets:
            continue
        anchors, absent = anchors_for(question.query, frequency, len(documents))
        evidence = QuestionEvidence(
            question_id=question.id, query=question.query, anchors=anchors, absent=absent
        )
        seen: set[str] = set()
        for term, _ in anchors:
            for document in documents:
                for candidate in _passages(document, term, context, per_term):
                    key = " ".join(candidate.text.split())[:80]
                    if key in seen:
                        continue
                    seen.add(key)
                    evidence.candidates.append(candidate)
            if len(evidence.candidates) >= max_candidates:
                break
        evidence.candidates = evidence.candidates[:max_candidates]
        results.append(evidence)
    return results
