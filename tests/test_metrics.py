from chunklab.eval.metrics import hit_rate_at_k, mrr, precision_at_k, recall_at_k
from chunklab.models import Chunk, QuestionResult, RetrievedChunk


def qr(hit, first_rank, found, total, hit_ranks=()):
    retrieved = []
    for i in range(5):
        c = Chunk(
            id=f"d:s:{i}", doc_id="d", text="x", token_count=1, char_span=(0, 1), strategy="s"
        )
        retrieved.append(
            RetrievedChunk(chunk=c, score=0.5, rank=i + 1, is_hit=(i + 1) in hit_ranks)
        )
    return QuestionResult(
        question_id="q",
        strategy="s",
        retrieved=retrieved,
        hit=hit,
        first_hit_rank=first_rank,
        gold_found_count=found,
        gold_total=total,
    )


# Fixture: q1 hit at rank 1 (1/1 gold, 2 hit chunks), q2 hit at rank 2 (1/2 gold),
# q3 miss (0/1).
RESULTS = [
    qr(True, 1, 1, 1, hit_ranks=(1, 3)),
    qr(True, 2, 1, 2, hit_ranks=(2,)),
    qr(False, None, 0, 1),
]


def test_hit_rate():
    assert hit_rate_at_k(RESULTS) == 2 / 3


def test_recall():
    assert recall_at_k(RESULTS) == (1.0 + 0.5 + 0.0) / 3


def test_mrr():
    assert mrr(RESULTS) == (1.0 + 0.5 + 0.0) / 3


def test_precision():
    assert precision_at_k(RESULTS, 5) == (2 / 5 + 1 / 5 + 0) / 3


def _short_result(n_retrieved: int, hit_ranks=()) -> QuestionResult:
    """A question whose retriever could only return `n_retrieved` chunks."""
    retrieved = [
        RetrievedChunk(
            chunk=Chunk(
                id=f"d:s:{i}", doc_id="d", text="x", token_count=1, char_span=(0, 1), strategy="s"
            ),
            score=0.5,
            rank=i + 1,
            is_hit=(i + 1) in hit_ranks,
        )
        for i in range(n_retrieved)
    ]
    return QuestionResult(
        question_id="q",
        strategy="s",
        retrieved=retrieved,
        hit=bool(hit_ranks),
        gold_found_count=1 if hit_ranks else 0,
        gold_total=1,
    )


def test_precision_divides_by_what_was_retrieved_not_by_k():
    """A retriever returns min(k, len(chunks)). On a corpus with fewer than k
    chunks the fixed denominator scored a perfect result as a miss: three
    chunks retrieved, all three relevant, reported as 0.60 at k=5. That is the
    one-page corpus a new user tries first."""
    perfect = _short_result(3, hit_ranks=(1, 2, 3))

    assert precision_at_k([perfect], 5) == 1.0


def test_precision_is_unchanged_when_the_corpus_is_large_enough():
    """The common case must not move: with k chunks returned the denominator is
    k either way."""
    assert precision_at_k(RESULTS, 5) == (2 / 5 + 1 / 5 + 0) / 3


def test_precision_handles_a_question_with_nothing_retrieved():
    assert precision_at_k([_short_result(0)], 5) == 0.0


def test_empty():
    assert hit_rate_at_k([]) == 0.0
    assert recall_at_k([]) == 0.0
    assert mrr([]) == 0.0
    assert precision_at_k([], 5) == 0.0


def test_retrieved_tokens_at_k():
    from chunklab.eval.metrics import retrieved_tokens_at_k

    # each result retrieves 5 chunks of 1 token -> 5 per question
    assert retrieved_tokens_at_k(RESULTS) == 5.0
    assert retrieved_tokens_at_k([]) == 0.0


def test_context_efficiency_hand_computed():
    from chunklab.eval.metrics import context_efficiency

    r = qr(True, 1, 1, 1, hit_ranks=(1,))
    r.found_gold_indices = [0]
    gold_tokens = {"q": [2]}  # found gold worth 2 tokens, retrieved 5 tokens
    assert context_efficiency([r], gold_tokens) == 2 / 5

    r_miss = qr(False, None, 0, 1)
    assert context_efficiency([r_miss], {"q": [2]}) == 0.0


def test_balanced_scores_hand_computed():
    from chunklab.eval.metrics import balanced_scores

    recalls = {"lean": 0.80, "fat": 0.90}
    tokens = {"lean": 1000.0, "fat": 3000.0}
    out = balanced_scores(recalls, tokens, lambda_=0.05)
    assert out["lean"] == 0.80  # minimum pays no penalty
    assert abs(out["fat"] - (0.90 - 0.05 * 2.0)) < 1e-9  # 3x tokens -> penalty 2*lambda


def test_balanced_scores_zero_lambda_is_recall():
    from chunklab.eval.metrics import balanced_scores

    recalls = {"a": 0.7, "b": 0.9}
    tokens = {"a": 100.0, "b": 900.0}
    assert balanced_scores(recalls, tokens, lambda_=0.0) == recalls
