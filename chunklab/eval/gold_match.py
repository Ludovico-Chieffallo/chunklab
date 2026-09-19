"""Gold-snippet matching: does a retrieved chunk contain the answer? (spec §4.4)"""

import re

from rapidfuzz import fuzz

from chunklab.models import GoldSlot, Question, QuestionResult, RetrievedChunk

_WS_RE = re.compile(r"\s+")


def variants(slot: GoldSlot) -> list[str]:
    """The interchangeable passages that fill one gold slot.

    A plain string is a slot with a single variant, which is what every
    question set written before nested slots existed contains.
    """
    return [slot] if isinstance(slot, str) else list(slot)


def normalize(text: str) -> str:
    return _WS_RE.sub(" ", text.lower()).strip()


def snippet_in_text(snippet: str, text: str, fuzzy_threshold: float = 0.90) -> bool:
    g, c = normalize(snippet), normalize(text)
    if not g or not c:
        return False
    if g in c:
        return True
    # Fuzzy containment: best alignment of the snippet inside the chunk.
    return fuzz.partial_ratio(g, c) / 100.0 >= fuzzy_threshold


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
