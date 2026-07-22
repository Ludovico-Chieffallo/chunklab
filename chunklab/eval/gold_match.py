"""Gold-snippet matching: does a retrieved chunk contain the answer? (spec §4.4)"""

import re

from rapidfuzz import fuzz

from chunklab.models import Question, QuestionResult, RetrievedChunk

_WS_RE = re.compile(r"\s+")


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
    """Mark hits on the retrieved chunks and compute per-question outcomes."""
    gold_total = len(question.gold_snippets)
    found: set[int] = set()
    first_hit_rank: int | None = None

    for rc in retrieved:
        rc.is_hit = False
        for gi, gold in enumerate(question.gold_snippets):
            if snippet_in_text(gold, rc.chunk.text, fuzzy_threshold):
                rc.is_hit = True
                found.add(gi)
                if first_hit_rank is None:
                    first_hit_rank = rc.rank

    split_across = False
    if len(found) < gold_total:
        # Would any missing snippet be found by joining two adjacent retrieved chunks?
        missing = [g for i, g in enumerate(question.gold_snippets) if i not in found]
        pairs = [
            (a, b) for i, a in enumerate(retrieved) for b in retrieved[i + 1 :] if _adjacent(a, b)
        ]
        for gold in missing:
            for a, b in pairs:
                first, second = sorted((a, b), key=lambda rc: rc.chunk.char_span[0])
                joined = first.chunk.text + " " + second.chunk.text
                if snippet_in_text(gold, joined, fuzzy_threshold):
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
    )
