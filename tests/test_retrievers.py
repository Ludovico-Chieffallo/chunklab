"""BM25, hybrid fusion, and the strategy x retriever matrix (roadmap phase 5).

Comparing chunking strategies under a single retriever measures how well chunking
suits *that* retriever. These tests pin the two new retrievers and the matrix that
lets the two axes be compared against each other.
"""

import pytest

from chunklab.config import default_config
from chunklab.models import Chunk, Document, Question
from chunklab.retrieval.bm25 import BM25Retriever
from chunklab.retrieval.hybrid import HybridRetriever
from chunklab.retrieval.registry import make_retrievers
from chunklab.retrieval.tokenize import tokenize
from chunklab.runner import run_evaluation

CHUNKS = [
    "Invoices are payable within thirty days of the invoice date.",
    "Support acknowledges a production outage within one hour on business days.",
    "The rate limit is 300 requests per minute on the Starter plan.",
    "Audit entries are immutable and readable for 400 days after creation.",
]


def _chunks(texts: list[str]) -> list[Chunk]:
    return [
        Chunk(
            id=f"c{i}",
            doc_id="doc",
            text=text,
            char_span=(0, len(text)),
            token_count=len(text.split()),
            strategy="test",
            index=i,
        )
        for i, text in enumerate(texts)
    ]


# --- tokenization ---------------------------------------------------------------


def test_words_are_lowercased_and_punctuation_dropped():
    assert tokenize("Invoices are PAYABLE, within 30 days.") == [
        "invoices",
        "are",
        "payable",
        "within",
        "30",
        "days",
    ]


def test_cjk_becomes_character_bigrams():
    """A `\\w+` tokenizer emits one token per Japanese sentence, so BM25 matches
    nothing. Bigrams are the standard fallback for unsegmented scripts."""
    terms = tokenize("請求書は三十日以内")

    assert all(len(t) == 2 for t in terms)
    assert "請求" in terms and "三十" in terms


def test_single_cjk_character_survives():
    assert tokenize("日") == ["日"]


def test_mixed_scripts_keep_their_own_rules():
    terms = tokenize("The API 請求書 endpoint")
    assert "the" in terms and "api" in terms and "endpoint" in terms
    assert "請求" in terms


# --- BM25 -----------------------------------------------------------------------


def test_exact_term_wins():
    retriever = BM25Retriever(_chunks(CHUNKS))

    top = retriever.retrieve("how many requests per minute on Starter?", 1)

    assert "300 requests per minute" in top[0].chunk.text


def test_ranks_are_dense_and_ordered():
    retriever = BM25Retriever(_chunks(CHUNKS))

    hits = retriever.retrieve("invoice payable days", 3)

    assert [h.rank for h in hits] == [1, 2, 3]
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_query_with_no_shared_terms_scores_zero():
    retriever = BM25Retriever(_chunks(CHUNKS))

    hits = retriever.retrieve("zzzz qqqq", 2)

    assert all(h.score == 0.0 for h in hits)
    assert [h.chunk.id for h in hits] == ["c0", "c1"], "ties must fall back to chunk order"


def test_common_terms_carry_less_weight_than_rare_ones():
    """'days' appears in most chunks; 'immutable' in one."""
    retriever = BM25Retriever(_chunks(CHUNKS))

    assert retriever.retrieve("immutable days", 1)[0].chunk.text.startswith("Audit entries")


def test_top_k_larger_than_the_index_is_clamped():
    retriever = BM25Retriever(_chunks(CHUNKS))
    assert len(retriever.retrieve("invoice", 99)) == len(CHUNKS)


def test_bm25_is_deterministic():
    a = BM25Retriever(_chunks(CHUNKS)).retrieve("invoice payable", 4)
    b = BM25Retriever(_chunks(CHUNKS)).retrieve("invoice payable", 4)
    assert [h.chunk.id for h in a] == [h.chunk.id for h in b]


# --- hybrid fusion --------------------------------------------------------------


class FakeRanker:
    """Returns a fixed ranking, ignoring the query."""

    def __init__(self, chunks: list[Chunk], order: list[int]) -> None:
        self._ranked = [chunks[i] for i in order]

    def retrieve(self, query: str, top_k: int):
        from chunklab.models import RetrievedChunk

        return [
            RetrievedChunk(chunk=c, score=1.0 / (i + 1), rank=i + 1)
            for i, c in enumerate(self._ranked[:top_k])
        ]


def test_agreement_between_retrievers_wins():
    """A chunk ranked 2nd by both beats one ranked 1st by a single retriever."""
    chunks = _chunks(CHUNKS)
    a = FakeRanker(chunks, [0, 1, 2, 3])
    b = FakeRanker(chunks, [3, 1, 2, 0])

    fused = HybridRetriever([a, b]).retrieve("q", 1)

    assert fused[0].chunk.id == "c1"


def test_fusion_is_order_independent():
    chunks = _chunks(CHUNKS)
    a = FakeRanker(chunks, [0, 1, 2, 3])
    b = FakeRanker(chunks, [2, 3, 0, 1])

    forward = [h.chunk.id for h in HybridRetriever([a, b]).retrieve("q", 4)]
    backward = [h.chunk.id for h in HybridRetriever([b, a]).retrieve("q", 4)]

    assert forward == backward


def test_fusion_ranks_are_renumbered():
    chunks = _chunks(CHUNKS)
    fused = HybridRetriever([FakeRanker(chunks, [0, 1, 2, 3])]).retrieve("q", 3)
    assert [h.rank for h in fused] == [1, 2, 3]


def test_hybrid_needs_at_least_one_retriever():
    with pytest.raises(ValueError, match="at least one"):
        HybridRetriever([])


# --- registry -------------------------------------------------------------------


def test_unknown_mode_is_rejected(fake_embedder):
    with pytest.raises(ValueError, match="unknown retrieval mode"):
        make_retrievers(["telepathy"], _chunks(CHUNKS), fake_embedder)


def test_hybrid_reuses_the_indexes_it_fuses(fake_embedder):
    """Asking for all three must not build two dense indexes."""
    built = make_retrievers(["dense", "bm25", "hybrid"], _chunks(CHUNKS), fake_embedder)

    assert set(built) == {"dense", "bm25", "hybrid"}
    assert built["hybrid"].retrievers[0] is built["dense"]
    assert built["hybrid"].retrievers[1] is built["bm25"]


# --- the matrix -----------------------------------------------------------------


def _corpus_and_questions():
    text = "\n\n".join(CHUNKS)
    document = Document(id="doc", source_path="d.md", text=text, elements=[], metadata={})
    questions = [
        Question(
            id="q1",
            query="when are invoices payable?",
            gold_snippets=["Invoices are payable within thirty days"],
        ),
        Question(
            id="q2",
            query="what is the rate limit?",
            gold_snippets=["300 requests per minute on the Starter plan"],
        ),
    ]
    return [document], questions


def test_matrix_has_one_entry_per_pair():
    documents, questions = _corpus_and_questions()
    config = default_config()
    config.embedding.backend = "fake"
    config.retrieval.compare = ["dense", "bm25", "hybrid"]

    report = run_evaluation(documents, questions, config)

    pairs = {(r.strategy, r.retriever) for r in report.strategy_results}
    assert len(pairs) == len(report.strategy_results), "duplicate matrix cells"
    assert {r for _, r in pairs} == {"dense", "bm25", "hybrid"}
    assert report.corpus_summary["retrieval_modes"] == ["dense", "bm25", "hybrid"]


def test_single_mode_keeps_the_flat_report():
    documents, questions = _corpus_and_questions()
    config = default_config()
    config.embedding.backend = "fake"

    report = run_evaluation(documents, questions, config)

    assert {r.retriever for r in report.strategy_results} == {"dense"}
    # No "strategy + retriever" labels when there is nothing to disambiguate.
    assert " + dense" not in report.recommendation


def test_recommendation_names_the_retriever_in_a_matrix():
    documents, questions = _corpus_and_questions()
    config = default_config()
    config.embedding.backend = "fake"
    config.retrieval.compare = ["dense", "bm25"]

    report = run_evaluation(documents, questions, config)

    assert any(mode in report.recommendation for mode in ("dense", "bm25", "DENSE", "BM25"))


def test_mode_alone_is_used_when_compare_is_empty():
    config = default_config()
    config.retrieval.mode = "bm25"
    assert config.retrieval.modes == ["bm25"]


def test_compare_rejects_duplicates():
    from pydantic import ValidationError

    from chunklab.config import RetrievalConfig

    with pytest.raises(ValidationError, match="twice"):
        RetrievalConfig(compare=["dense", "dense"])
