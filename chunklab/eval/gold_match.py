"""Gold-snippet matching: does a retrieved chunk contain the answer? (spec §4.4)"""

import heapq
import re
from typing import NamedTuple

from rapidfuzz import fuzz

from chunklab.models import GoldSlot, Question, QuestionResult, RetrievedChunk

_WS_RE = re.compile(r"\s+")

#: A number as written: digits, with any decimal or thousands separators inside it.
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


def variants(slot: GoldSlot) -> list[str]:
    """The interchangeable passages that fill one gold slot.

    A plain string is a slot with a single variant, which is what every
    question set written before nested slots existed contains.
    """
    return [slot] if isinstance(slot, str) else list(slot)


def normalize(text: str) -> str:
    return _WS_RE.sub(" ", text.lower()).strip()


def numbers_in(text: str) -> list[str]:
    """The numbers in `text`, in order, exactly as written."""
    return _NUMBER_RE.findall(text)


def same_numbers(snippet: str, passage: str) -> bool:
    """True when every number in `snippet` appears in `passage`, in the same order.

    Extra numbers in the passage are fine: a window widened to whole words can
    pick one up at either edge.
    """
    remaining = iter(numbers_in(passage))
    # `in` advances the iterator, so each number is looked for after the previous one.
    return all(number in remaining for number in numbers_in(snippet))


def snap_to_words(text: str, start: int, end: int) -> tuple[int, int]:
    """Widen [start, end) so it never cuts a word in half."""
    while 0 < start < len(text) and not text[start - 1].isspace() and not text[start].isspace():
        start -= 1
    while 0 < end < len(text) and not text[end - 1].isspace() and not text[end].isspace():
        end += 1
    return start, end


class Match(NamedTuple):
    """Where a snippet aligns in a text: [start, end) and a 0-1 similarity."""

    score: float
    start: int
    end: int
    numbers_agree: bool


def _agrees(snippet: str, text: str, start: int, end: int) -> bool:
    start, end = snap_to_words(text, start, end)
    return same_numbers(snippet, text[start:end])


def exact_occurrences(snippet: str, text: str) -> list[int]:
    """Offsets of every occurrence of `snippet` in `text` that leaves its numbers whole.

    Plain containment finds "5 business days" inside "15 business days". Both
    arguments are normalized text.
    """
    found: list[int] = []
    at = text.find(snippet)
    while at != -1:
        if _agrees(snippet, text, at, at + len(snippet)):
            found.append(at)
        at = text.find(snippet, at + 1)
    return found


def locate(snippet: str, text: str, fuzzy_threshold: float) -> Match | None:
    """The passage of `text` that counts as `snippet`, or else the closest one.

    Both arguments are normalized. An exact occurrence wins; otherwise fuzzy
    alignments are tried best first. A passage whose numbers differ from the
    snippet's never counts, however close its words: "within 14 days" is not
    "within 30 days", and typo tolerance must not say it is. On templated text the
    closest alignment is often exactly that, so it is set aside and the rest of
    the text is searched again.

    When nothing counts, the closest alignment is returned all the same, with its
    score and `numbers_agree`, so `validate` can show it and say why it failed.
    None only when there is nothing to align.
    """
    if not snippet or not text:
        return None
    exact = exact_occurrences(snippet, text)
    if exact:
        return Match(1.0, exact[0], exact[0] + len(snippet), True)

    pending: list[tuple[float, int, int, int, int]] = []

    def search(lo: int, hi: int) -> None:
        if hi - lo < len(snippet):
            # Too short to contain the snippet. partial_ratio would turn the roles around
            # and find the text inside the snippet, so any fragment of the answer would
            # score 100: compare the two whole instead.
            heapq.heappush(pending, (-fuzz.ratio(snippet, text[lo:hi]), lo, lo, hi, hi))
            return
        alignment = fuzz.partial_ratio_alignment(snippet, text[lo:hi])
        if alignment is not None:
            item = (-alignment.score, lo, lo + alignment.dest_start, lo + alignment.dest_end, hi)
            heapq.heappush(pending, item)

    search(0, len(text))
    closest: Match | None = None
    while pending:
        negative_score, lo, start, end, hi = heapq.heappop(pending)
        match = Match(-negative_score / 100.0, start, end, _agrees(snippet, text, start, end))
        if closest is None:
            closest = match
        if match.score < fuzzy_threshold:
            break
        if match.numbers_agree:
            return match
        # The same words with other numbers: not this snippet. Search either side of it,
        # where there is still room for the whole snippet.
        start, end = snap_to_words(text, start, end)
        if end <= start:
            continue
        for side_lo, side_hi in ((lo, start), (end, hi)):
            if side_hi - side_lo >= len(snippet):
                search(side_lo, side_hi)
    return closest


def accepted(match: Match | None, fuzzy_threshold: float) -> bool:
    """Whether scoring counts `match` as the snippet."""
    return match is not None and match.score >= fuzzy_threshold and match.numbers_agree


def snippet_in_text(snippet: str, text: str, fuzzy_threshold: float = 0.90) -> bool:
    match = locate(normalize(snippet), normalize(text), fuzzy_threshold)
    return accepted(match, fuzzy_threshold)


def _adjacent(a: RetrievedChunk, b: RetrievedChunk) -> bool:
    """True when the two chunks are contiguous (or overlapping) in the source."""
    if a.chunk.doc_id != b.chunk.doc_id:
        return False
    (s1, e1), (s2, e2) = a.chunk.char_span, b.chunk.char_span
    if s1 > s2:
        (s1, e1), (s2, e2) = (s2, e2), (s1, e1)
    return s2 <= e1 + 1


def score_question(
    question: Question,
    retrieved: list[RetrievedChunk],
    strategy: str,
    fuzzy_threshold: float = 0.90,
) -> QuestionResult:
    """Mark hits on the retrieved chunks and compute per-question outcomes.

    Each entry of `gold_snippets` is a slot, and recall is the fraction of
    slots filled. A slot written as a list is filled by *any* of its variants:
    on a corpus where several documents answer the same question, that is the
    only way to annotate all the places the answer lives without the extra
    annotations counting against the score.
    """
    slots = [variants(slot) for slot in question.gold_snippets]
    gold_total = len(slots)
    found: set[int] = set()
    first_hit_rank: int | None = None

    for rc in retrieved:
        rc.is_hit = False
        for gi, slot in enumerate(slots):
            if any(snippet_in_text(gold, rc.chunk.text, fuzzy_threshold) for gold in slot):
                rc.is_hit = True
                found.add(gi)
                if first_hit_rank is None:
                    first_hit_rank = rc.rank

    split_across = False
    if len(found) < gold_total:
        # Would any unfilled slot be filled by joining two adjacent retrieved chunks?
        missing = [slot for i, slot in enumerate(slots) if i not in found]
        pairs = [
            (a, b) for i, a in enumerate(retrieved) for b in retrieved[i + 1 :] if _adjacent(a, b)
        ]
        for slot in missing:
            for a, b in pairs:
                first, second = sorted((a, b), key=lambda rc: rc.chunk.char_span[0])
                joined = first.chunk.text + " " + second.chunk.text
                if any(snippet_in_text(gold, joined, fuzzy_threshold) for gold in slot):
                    split_across = True
                    break
            if split_across:
                break

    return QuestionResult(
        question_id=question.id,
        strategy=strategy,
        retrieved=retrieved,
        hit=bool(found),
        first_hit_rank=first_hit_rank,
        split_across_chunks=split_across,
        gold_found_count=len(found),
        gold_total=gold_total,
        found_gold_indices=sorted(found),
    )
