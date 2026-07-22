from chunklab.eval.gold_match import score_question, snippet_in_text
from chunklab.models import Chunk, Question, RetrievedChunk


def make_chunk(text: str, span: tuple[int, int], idx: int = 0) -> Chunk:
    return Chunk(
        id=f"d:s:{idx}",
        doc_id="d",
        text=text,
        token_count=len(text.split()),
        char_span=span,
        strategy="s",
    )


def make_retrieved(texts_spans: list[tuple[str, tuple[int, int]]]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk=make_chunk(t, sp, i), score=1.0 - i * 0.1, rank=i + 1)
        for i, (t, sp) in enumerate(texts_spans)
    ]


def test_exact_containment():
    assert snippet_in_text("paid at 1.5x the rate", "Overtime is PAID at  1.5x\nthe rate today")


def test_fuzzy_containment_near_threshold():
    # one small typo inside a longer chunk
    assert snippet_in_text(
        "written notice at least 30 days prior to termination",
        "either party gives writen notice at least 30 days prior to termination of contract",
        fuzzy_threshold=0.90,
    )


def test_no_hit_below_threshold():
    assert not snippet_in_text(
        "written notice at least 30 days prior to termination",
        "the cafeteria serves lunch between noon and two",
    )


def test_score_question_hit_and_rank():
    q = Question(id="q", query="?", gold_snippets=["the answer is forty-two"])
    retrieved = make_retrieved(
        [
            ("nothing relevant here", (0, 20)),
            ("indeed, the answer is forty-two, as stated", (100, 140)),
        ]
    )
    r = score_question(q, retrieved, "s")
    assert r.hit and r.first_hit_rank == 2
    assert r.gold_found_count == 1 and r.gold_total == 1
    assert not r.split_across_chunks
    assert retrieved[1].is_hit and not retrieved[0].is_hit


def test_split_across_adjacent_chunks_detected():
    q = Question(id="q", query="?", gold_snippets=["alpha beta gamma delta"])
    # gold severed across two adjacent chunks (spans touch at 30)
    retrieved = make_retrieved(
        [
            ("intro text alpha beta", (0, 30)),
            ("gamma delta and the rest", (30, 60)),
        ]
    )
    r = score_question(q, retrieved, "s")
    assert not r.hit
    assert r.split_across_chunks


def test_split_not_flagged_for_distant_chunks():
    q = Question(id="q", query="?", gold_snippets=["alpha beta gamma delta"])
    retrieved = make_retrieved(
        [
            ("intro text alpha beta", (0, 30)),
            ("gamma delta and the rest", (500, 540)),  # not adjacent
        ]
    )
    r = score_question(q, retrieved, "s")
    assert not r.hit
    assert not r.split_across_chunks
