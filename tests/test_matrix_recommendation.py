"""With a strategy x retriever matrix, the recommendation must answer both axes.

Found on a real run — 26 CRISPR review papers, 20 scored questions,
`--compare-retrievers`. The two best cells of the matrix were
`recursive + hybrid` and `recursive + bm25`: the *same chunker*. The report
duly declared "no winner" and asked for 109 more questions, so on the one
question chunklab exists to answer it said nothing at all — while under that
same retriever `recursive` beat `semantic` by +0.325 recall with a 95% CI of
[+0.150, +0.525]. The run had decided the chunking question and the tool hid it
behind a comparison of two retrievers.

The fix is to stop ranking cells against each other. A matrix gets one gated
verdict per axis: which chunker (holding the retriever fixed) and which
retriever (holding the chunker fixed). Each held value is the best performer on
its own axis, which is itself a choice made from these data — so the verdict
says so rather than presenting the pairing as a general fact.
"""

from chunklab.config import default_config
from chunklab.models import ChunkHealth, QuestionResult, StrategyResult
from chunklab.runner import _build_recommendation

N = 20


def _hits(indices) -> list[float]:
    """Per-question recall vector: 1.0 on `indices`, 0.0 elsewhere."""
    wanted = set(indices)
    return [1.0 if i in wanted else 0.0 for i in range(N)]


def _cell(strategy: str, retriever: str, scores: list[float], tokens: float = 2000.0):
    health = ChunkHealth(
        num_chunks=10,
        tokens_min=100,
        tokens_median=400,
        tokens_mean=400,
        tokens_max=800,
        pct_tiny=0.0,
        pct_oversized=0.0,
        boundary_health=1.0,
    )
    mean = sum(scores) / len(scores)
    return StrategyResult(
        strategy=strategy,
        retriever=retriever,
        recall_at_k=mean,
        hit_rate_at_k=mean,
        mrr=mean,
        precision_at_k=mean,
        retrieved_tokens_at_k=tokens,
        balanced_score=mean,
        chunk_health=health,
        per_question=[
            QuestionResult(
                question_id=f"q{i}",
                strategy=strategy,
                retrieved=[],
                hit=score > 0,
                gold_found_count=int(score),
                gold_total=1,
            )
            for i, score in enumerate(scores)
        ],
    )


def _real_run_matrix() -> list[StrategyResult]:
    """The shape of the CRISPR run, reduced to the two strategies that matter.

    `hybrid` and `bm25` differ on five questions under `recursive` — three one
    way, two the other — so the top two cells cannot be separated. Under
    `hybrid`, `recursive` beats `semantic` on nine questions and loses none, so
    the chunking axis is decided.
    """
    cells = [
        _cell("recursive", "hybrid", _hits(range(13))),  # 0.65
        _cell("recursive", "bm25", _hits([*range(10), 13, 14])),  # 0.60
        _cell("recursive", "dense", _hits(range(8))),  # 0.40
        _cell("semantic", "hybrid", _hits(range(4))),  # 0.20
        _cell("semantic", "bm25", _hits(range(4))),  # 0.20
        _cell("semantic", "dense", _hits(range(3))),  # 0.15
    ]
    # The runner ranks cells before handing them over; mirror that here.
    return sorted(cells, key=lambda r: -r.recall_at_k)


def _blocks(text: str) -> dict[str, str]:
    """The recommendation split into its per-axis blocks, keyed by heading."""
    blocks = {}
    for part in text.split("\n\n"):
        heading, _, body = part.partition(" — ")
        blocks[heading] = body
    return blocks


def test_matrix_recommendation_decides_the_chunking_axis():
    """The regression: a tie between two cells of one chunker must not hide a
    decided comparison between chunkers."""
    text = _build_recommendation(_real_run_matrix(), default_config(), num_scored=N)
    chunking = _blocks(text)["Chunking"]

    assert "RECURSIVE" in chunking
    assert "indistinguishable" not in chunking
    # The verdict has to name what it beat, or "best" is unfalsifiable.
    assert "semantic" in chunking


def test_matrix_recommendation_declares_the_tie_on_the_retrieval_axis():
    text = _build_recommendation(_real_run_matrix(), default_config(), num_scored=N)
    retrieval = _blocks(text)["Retrieval"]

    assert "indistinguishable" in retrieval
    assert "hybrid" in retrieval and "bm25" in retrieval
    # Advice on the retrieval axis must not tell the reader to pick a chunker.
    assert "committing to a strategy" not in retrieval


def test_matrix_recommendation_names_and_qualifies_the_held_axis():
    """Each verdict holds the other axis at its best performer — which was
    chosen from the same data, so the text must not present it as given."""
    text = _build_recommendation(_real_run_matrix(), default_config(), num_scored=N)
    blocks = _blocks(text)

    assert "'hybrid'" in blocks["Chunking"]
    assert "'recursive'" in blocks["Retrieval"]
    for body in (blocks["Chunking"], blocks["Retrieval"]):
        assert "chosen from these data" in body


def test_matrix_verdicts_compare_like_with_like():
    """Neither axis may compare a competitor against itself — the defect that
    produced 'recursive + hybrid vs recursive + bm25'."""
    text = _build_recommendation(_real_run_matrix(), default_config(), num_scored=N)
    blocks = _blocks(text)

    # The chunking verdict talks about chunkers; the retrieval one about retrievers.
    assert "semantic" in blocks["Chunking"]
    assert "semantic" not in blocks["Retrieval"]
    assert "bm25" in blocks["Retrieval"]
    assert "bm25" not in blocks["Chunking"].replace("'hybrid'", "")


def test_single_retriever_keeps_one_unlabelled_block():
    """Without a matrix there is only one axis, so the headings would be noise."""
    ranked = [c for c in _real_run_matrix() if c.retriever == "hybrid"]
    text = _build_recommendation(ranked, default_config(), num_scored=N)

    assert "Chunking — " not in text
    assert "Retrieval — " not in text
    assert "\n\n" not in text
    assert "RECURSIVE" in text


def test_winner_states_its_margin_over_the_runner_up():
    """'best' with no margin and no interval is exactly the unfalsifiable claim
    this tool exists to refuse."""
    ranked = [c for c in _real_run_matrix() if c.retriever == "hybrid"]
    text = _build_recommendation(ranked, default_config(), num_scored=N)

    assert "95% CI" in text
    assert "+0.4" in text  # recursive 0.65 - semantic 0.20
