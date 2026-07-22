"""Chunk-health diagnostics, independent of retrieval (spec §4.6)."""

import statistics

from chunklab.models import Chunk, ChunkHealth, Document
from chunklab.text_utils import sentence_spans


def _boundary_health(chunks: list[Chunk], documents: dict[str, Document]) -> float:
    """% of chunks that neither start nor end mid-sentence."""
    doc_sentences = {doc_id: sentence_spans(doc.text) for doc_id, doc in documents.items()}
    clean = 0
    for chunk in chunks:
        spans = doc_sentences.get(chunk.doc_id, [])
        if not spans:
            clean += 1
            continue
        start, end = chunk.char_span
        # Clean start: no sentence straddles the start boundary; same for the end.
        starts_mid = any(s < start < e for s, e in spans)
        ends_mid = any(s < end < e for s, e in spans)
        if not starts_mid and not ends_mid:
            clean += 1
    return clean / len(chunks)


def _table_integrity(chunks: list[Chunk], documents: dict[str, Document]) -> float | None:
    """% of source tables fully contained in a single chunk. None if no tables."""
    tables = [
        (doc_id, el.char_span)
        for doc_id, doc in documents.items()
        for el in doc.elements
        if el.type == "table"
    ]
    if not tables:
        return None
    intact = 0
    for doc_id, (ts, te) in tables:
        if any(
            c.doc_id == doc_id and c.char_span[0] <= ts and c.char_span[1] >= te for c in chunks
        ):
            intact += 1
    return intact / len(tables)


def compute_chunk_health(
    chunks: list[Chunk],
    documents: dict[str, Document],
    min_floor_tokens: int = 200,
    max_tokens: int = 1000,
    embedder_max_seq: int | None = None,
    histogram_bucket: int = 100,
) -> ChunkHealth:
    if not chunks:
        raise ValueError("cannot compute health of zero chunks")
    tokens = [c.token_count for c in chunks]
    oversize_limit = max_tokens
    if embedder_max_seq is not None:
        oversize_limit = min(oversize_limit, embedder_max_seq)

    histogram: dict[int, int] = {}
    for t in tokens:
        bucket = (t // histogram_bucket) * histogram_bucket
        histogram[bucket] = histogram.get(bucket, 0) + 1

    return ChunkHealth(
        num_chunks=len(chunks),
        tokens_min=min(tokens),
        tokens_median=statistics.median(tokens),
        tokens_mean=statistics.mean(tokens),
        tokens_max=max(tokens),
        pct_tiny=sum(1 for t in tokens if t < min_floor_tokens) / len(tokens),
        pct_oversized=sum(1 for t in tokens if t > oversize_limit) / len(tokens),
        boundary_health=_boundary_health(chunks, documents),
        table_integrity=_table_integrity(chunks, documents),
        token_histogram=sorted(histogram.items()),
    )
