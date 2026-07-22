"""Chunk-boundary visualization data for the HTML report.

For one sample document and each strategy, produce the chunk boundaries plus
where each gold snippet landed (inside one chunk = clean, straddling a
boundary = split, not found = missing). Positions are char offsets, which the
template renders as percentage widths.
"""

from pydantic import BaseModel

from chunklab.eval.gold_match import normalize
from chunklab.models import Chunk, Document, Question


class GoldMarker(BaseModel):
    question_id: str
    start: int
    end: int
    status: str  # "clean" | "split" | "missing"


class StrategyViz(BaseModel):
    strategy: str
    boundaries: list[int]  # chunk start offsets (plus final end)
    gold_markers: list[GoldMarker]


class DocViz(BaseModel):
    doc_id: str
    doc_length: int
    strategies: list[StrategyViz]


def _locate_gold(doc_text: str, snippet: str) -> tuple[int, int] | None:
    """Find the gold snippet's span in the source, whitespace-insensitively."""
    idx = doc_text.find(snippet)
    if idx != -1:
        return idx, idx + len(snippet)
    # Normalize both and map the normalized position back approximately.
    norm_doc = normalize(doc_text)
    norm_snip = normalize(snippet)
    nidx = norm_doc.find(norm_snip)
    if nidx == -1:
        return None
    # Walk the original text accumulating normalized chars to find the offset.
    count = 0
    start = None
    in_ws = True
    for i, ch in enumerate(doc_text):
        is_ws = ch.isspace()
        if is_ws and in_ws:
            continue
        count += 1
        in_ws = is_ws
        if start is None and count > nidx:
            start = i
        if count >= nidx + len(norm_snip):
            return (start if start is not None else i, i + 1)
    return None


def build_doc_viz(
    document: Document,
    questions: list[Question],
    chunks_by_strategy: dict[str, list[Chunk]],
) -> DocViz:
    strategies: list[StrategyViz] = []
    for strategy, chunks in chunks_by_strategy.items():
        doc_chunks = sorted(
            (c for c in chunks if c.doc_id == document.id), key=lambda c: c.char_span[0]
        )
        boundaries = [c.char_span[0] for c in doc_chunks]
        if doc_chunks:
            boundaries.append(doc_chunks[-1].char_span[1])

        markers: list[GoldMarker] = []
        for q in questions:
            for gold in q.gold_snippets:
                span = _locate_gold(document.text, gold)
                if span is None:
                    continue
                gs, ge = span
                inside_one = any(c.char_span[0] <= gs and c.char_span[1] >= ge for c in doc_chunks)
                status = "clean" if inside_one else "split"
                markers.append(GoldMarker(question_id=q.id, start=gs, end=ge, status=status))
        strategies.append(
            StrategyViz(strategy=strategy, boundaries=boundaries, gold_markers=markers)
        )
    return DocViz(doc_id=document.id, doc_length=len(document.text), strategies=strategies)
