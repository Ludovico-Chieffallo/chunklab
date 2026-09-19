"""Validate a question set against a corpus before evaluating it (roadmap 3.1).

Writing gold snippets by hand is the main source of friction in chunklab, and
a snippet that drifted by one word silently scores zero for every strategy —
which looks like a chunking problem and is not. `chunklab validate` finds
those cases and prints the corrected verbatim text, ready to paste.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from chunklab.chunkers.base import section_path_at
from chunklab.models import Document, Question
from chunklab.text_utils import count_tokens

#: Gold snippets shorter than this match by accident, and the odds rise sharply
#: with chunk size: measured with the default 0.90 fuzzy threshold, the 2-token
#: 'annual report' hit an unrelated 50-word chunk 12% of the time and an
#: unrelated 1500-word chunk 97% of the time, while a 12-token sentence never
#: hit at any size. That is a bias toward whichever strategy makes the biggest
#: chunks, on top of the containment bias `balanced` already prices in.
MIN_GOLD_TOKENS = 5

#: Headings under which text is a list of works cited rather than prose.
BIBLIOGRAPHY_HEADING = re.compile(
    r"^\s*(?:\d+[.)]?\s*)?(references|bibliography|works cited|literature cited|reference list)\b",
    re.IGNORECASE,
)

Severity = Literal["error", "warning"]


class Issue(BaseModel):
    severity: Severity
    kind: str
    question_id: str
    message: str
    suggestion: str | None = None  # verbatim replacement, ready to paste
    location: str | None = None  # "doc_id:char_offset"
    similarity: float | None = None


class ValidationReport(BaseModel):
    issues: list[Issue] = Field(default_factory=list)
    num_questions: int = 0
    num_scored: int = 0
    num_gold_snippets: int = 0

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Normalize like `gold_match.normalize` while tracking original offsets.

    Returns the normalized text and, for each normalized character, the offset
    of the original character it came from.
    """
    out: list[str] = []
    offsets: list[int] = []
    prev_space = True  # leading whitespace is dropped
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            offsets.append(i)
            prev_space = True
        else:
            out.append(ch.lower())
            offsets.append(i)
            prev_space = False
    while out and out[-1] == " ":
        out.pop()
        offsets.pop()
    return "".join(out), offsets


class _IndexedDoc:
    def __init__(self, document: Document) -> None:
        self.document = document
        self.norm, self.offsets = normalize_with_map(document.text)

    def occurrences(self, norm_gold: str) -> list[int]:
        """Offsets into the original text of every occurrence of `norm_gold`."""
        found: list[int] = []
        at = self.norm.find(norm_gold)
        while at != -1:
            found.append(self.offsets[at])
            at = self.norm.find(norm_gold, at + 1)
        return found


def in_bibliography(document: Document, offset: int) -> bool:
    """True when `offset` sits anywhere under a reference-list heading.

    The whole heading trail is tested, not just the innermost heading. PDF
    converters routinely promote a running page header to a heading, and one
    of those landing inside a reference list would otherwise mask it: on the
    corpus this check was built from, eight of nine occurrences of a cited
    paper title resolved to 'References' and the ninth to the journal name
    printed at the top of the page.
    """
    return any(BIBLIOGRAPHY_HEADING.match(heading) for heading in section_path_at(document, offset))


#: Below this similarity the closest window is noise, so no fix is proposed.
SUGGESTION_FLOOR = 0.60


def _snap_to_words(text: str, start: int, end: int) -> tuple[int, int]:
    """Widen [start, end) so it never cuts a word in half."""
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    while end < len(text) and not text[end - 1].isspace() and not text[end].isspace():
        end += 1
    return start, end


def _best_match(gold: str, docs: list[_IndexedDoc]) -> tuple[float, str | None, str | None]:
    """Best fuzzy alignment of `gold` across docs.

    Returns (similarity 0-1, verbatim text of the aligned window, "doc:offset").
    The text is None when nothing plausible was found, so the caller never
    prints a misleading "correction".
    """
    norm_gold, _ = normalize_with_map(gold)
    best: tuple[float, str | None, str | None] = (0.0, None, None)
    for d in docs:
        if not d.norm or not norm_gold:
            continue
        alignment = fuzz.partial_ratio_alignment(norm_gold, d.norm)
        if alignment is None:
            continue
        score = alignment.score / 100.0
        if score <= best[0]:
            continue
        start_n, end_n = alignment.dest_start, alignment.dest_end
        if start_n >= len(d.offsets):
            continue
        start = d.offsets[start_n]
        end = d.offsets[min(end_n, len(d.offsets) - 1)] + 1
        start, end = _snap_to_words(d.document.text, start, end)
        best = (score, d.document.text[start:end].strip(), f"{d.document.id}:{start}")

    if best[0] < SUGGESTION_FLOOR:
        return (best[0], None, None)
    return best


def validate_questions(
    questions: list[Question],
    documents: list[Document],
    fuzzy_threshold: float = 0.90,
    min_gold_tokens: int = MIN_GOLD_TOKENS,
) -> ValidationReport:
    """Check a question set against the corpus it will be evaluated on."""
    report = ValidationReport(num_questions=len(questions))
    indexed = [_IndexedDoc(d) for d in documents]

    seen: dict[str, int] = {}
    for q in questions:
        seen[q.id] = seen.get(q.id, 0) + 1
    for qid, count in seen.items():
        if count > 1:
            report.issues.append(
                Issue(
                    severity="error",
                    kind="duplicate_id",
                    question_id=qid,
                    message=f"question id '{qid}' appears {count} times; ids must be unique",
                )
            )

    for q in questions:
        if not q.gold_snippets:
            report.issues.append(
                Issue(
                    severity="warning",
                    kind="no_gold",
                    question_id=q.id,
                    message="no gold_snippets: this question is skipped when scoring",
                )
            )
            continue

        report.num_scored += 1
        for gold in q.gold_snippets:
            report.num_gold_snippets += 1
            norm_gold, _ = normalize_with_map(gold)

            if not norm_gold:
                report.issues.append(
                    Issue(
                        severity="error",
                        kind="empty_gold",
                        question_id=q.id,
                        message="gold snippet is empty",
                    )
                )
                continue

            containing = [d for d in indexed if norm_gold in d.norm]
            if not containing:
                score, verbatim, location = _best_match(gold, indexed)
                if score >= fuzzy_threshold:
                    report.issues.append(
                        Issue(
                            severity="warning",
                            kind="fuzzy_only",
                            question_id=q.id,
                            message=(
                                f"gold snippet matches only fuzzily ({score:.0%}); scoring "
                                "will rely on the fuzzy threshold. Replace with the "
                                "verbatim source text."
                            ),
                            suggestion=verbatim,
                            location=location,
                            similarity=score,
                        )
                    )
                else:
                    detail = (
                        f"closest match {score:.0%}"
                        if verbatim
                        else "no similar passage in the corpus - wrong document, or the "
                        "snippet was paraphrased rather than copied"
                    )
                    report.issues.append(
                        Issue(
                            severity="error",
                            kind="not_found",
                            question_id=q.id,
                            message=(
                                f"gold snippet not found in the corpus ({detail}): {gold[:70]!r}"
                            ),
                            suggestion=verbatim,
                            location=location,
                            similarity=score,
                        )
                    )
                continue

            # A snippet that only ever appears in a reference list is a cited
            # paper title, not evidence. It corrupts scoring in both directions:
            # one such snippet was "found" by all fifteen cells of a real matrix
            # because every strategy retrieves some bibliography chunk, and
            # another by none, because a list of citations is nowhere near the
            # query. Neither outcome says anything about chunking.
            placements = [(d, off) for d in containing for off in d.occurrences(norm_gold)]
            if placements and all(in_bibliography(d.document, off) for d, off in placements):
                first_doc, first_off = placements[0]
                report.issues.append(
                    Issue(
                        severity="error",
                        kind="citation_only",
                        question_id=q.id,
                        message=(
                            f"gold snippet occurs only inside reference lists "
                            f"({len(placements)} time(s) across {len(containing)} document(s)); "
                            "this is a cited work's title, not a passage that answers the "
                            "question. Replace it with the sentence in the body that states "
                            "the finding."
                        ),
                        location=f"{first_doc.document.id}:{first_off}",
                    )
                )
                continue

            if len(containing) > 1:
                where = ", ".join(d.document.id for d in containing[:4])
                report.issues.append(
                    Issue(
                        severity="warning",
                        kind="ambiguous",
                        question_id=q.id,
                        message=(
                            f"gold snippet appears in {len(containing)} documents ({where}); "
                            "any of them counts as a hit, which may not be what you mean"
                        ),
                    )
                )

            if count_tokens(gold) < min_gold_tokens:
                report.issues.append(
                    Issue(
                        severity="warning",
                        kind="too_short",
                        question_id=q.id,
                        message=(
                            f"gold snippet is only {count_tokens(gold)} tokens; short "
                            "snippets match by accident and inflate every strategy"
                        ),
                    )
                )

    return report
