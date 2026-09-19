"""A gold slot may be filled by any of several interchangeable passages.

Found on a run over 26 CRISPR review papers, where the same facts are stated
in most of them. Asked "What DNA sequence requirement is necessary for Cas9
target recognition?", every strategy retrieved a section headed `### PAM`
reading "The protospacer-adjacent motif (PAM) is strictly required to be
immediately next to the 3' end of the target sequence… the PAM is typically
NGG" — a better answer than the annotated gold — and scored zero, because the
annotation named one passage in one paper.

The obvious response, annotating every place the answer appears, made it
worse: `gold_snippets` was conjunctive, so recall was found/total and three
equally valid passages with one retrieved scored 0.33. The tool punished the
honest annotation.

Each entry is now a slot. A slot written as a list is filled by any of its
variants, so alternatives raise the ceiling instead of the denominator.
"""

from chunklab.eval.gold_match import score_question, variants
from chunklab.models import Chunk, Question, RetrievedChunk

PAM_VARIANTS = [
    "must be immediately adjacent to the NGG motif",
    "PAM is strictly required to be immediately next to the 3' end",
    "the PAM is typically NGG",
]


def _retrieved(*texts: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=Chunk(
                id=f"d:s:{i}",
                doc_id="d",
                text=text,
                token_count=len(text.split()),
                char_span=(i * 1000, i * 1000 + len(text)),
                strategy="s",
            ),
            score=0.9,
            rank=i + 1,
        )
        for i, text in enumerate(texts)
    ]


def _score(gold_snippets, *texts):
    question = Question(id="q", query="what does Cas9 need?", gold_snippets=gold_snippets)
    return score_question(question, _retrieved(*texts), "s")


# --- the helper -----------------------------------------------------------------


def test_a_plain_string_is_a_slot_with_one_variant():
    assert variants("only this") == ["only this"]


def test_a_list_is_a_slot_with_several_variants():
    assert variants(["a", "b"]) == ["a", "b"]


# --- disjunctive slots ----------------------------------------------------------


def test_any_variant_fills_the_slot():
    for variant in PAM_VARIANTS:
        result = _score([PAM_VARIANTS], f"Background text. {variant}. More text.")
        assert result.gold_found_count == 1, variant
        assert result.gold_total == 1
        assert result.hit


def test_gold_total_counts_slots_not_variants():
    """The regression: three alternatives used to mean three requirements."""
    result = _score([PAM_VARIANTS], f"A chunk saying {PAM_VARIANTS[1]} and nothing else.")

    assert result.gold_total == 1
    assert result.gold_found_count / result.gold_total == 1.0


def test_adding_alternatives_never_lowers_the_score():
    """The defect that motivated slots, stated as an invariant."""
    text = f"Background. {PAM_VARIANTS[0]}. More."
    one = _score([PAM_VARIANTS[0]], text)
    many = _score([PAM_VARIANTS], text)

    assert one.gold_found_count / one.gold_total == 1.0
    assert many.gold_found_count / many.gold_total == 1.0


def test_a_slot_no_variant_fills_stays_empty():
    result = _score([PAM_VARIANTS], "An unrelated passage about base editing efficiency.")

    assert result.gold_found_count == 0
    assert not result.hit


def test_required_and_alternative_slots_mix():
    """Conjunctive across slots, disjunctive within one."""
    gold = ["cytidine deaminase catalyzes the deamination of C into U", PAM_VARIANTS]

    both = _score(gold, "cytidine deaminase catalyzes the deamination of C into U", PAM_VARIANTS[2])
    assert (both.gold_found_count, both.gold_total) == (2, 2)

    one = _score(gold, "cytidine deaminase catalyzes the deamination of C into U")
    assert (one.gold_found_count, one.gold_total) == (1, 2)


def test_found_indices_point_at_slots():
    gold = ["first required passage here", PAM_VARIANTS]
    result = _score(gold, PAM_VARIANTS[0])

    assert result.found_gold_indices == [1]


def test_split_across_chunks_considers_every_variant():
    """A slot whose variant only appears across a chunk boundary is still the
    chunker severing an answer, and must be reported as such."""
    question = Question(id="q", query="what?", gold_snippets=[PAM_VARIANTS])
    first, second = _retrieved("Some lead in text, PAM is strictly required", "")
    second.chunk.text = "to be immediately next to the 3' end of the target."
    second.chunk.char_span = (len(first.chunk.text), len(first.chunk.text) + 60)

    result = score_question(question, [first, second], "s")

    assert result.gold_found_count == 0
    assert result.split_across_chunks


# --- nothing written before slots existed may change ----------------------------


def test_plain_string_lists_score_exactly_as_before():
    gold = ["first passage text here", "second passage text here"]

    both = _score(gold, "first passage text here", "second passage text here")
    assert (both.gold_found_count, both.gold_total) == (2, 2)

    half = _score(gold, "first passage text here")
    assert (half.gold_found_count, half.gold_total) == (1, 2)


def test_a_string_only_question_serialises_unchanged():
    """`questions_sha256` hashes the question's JSON, so widening the type must
    not move the hash of any question set already in use."""
    question = Question(id="q", query="q?", gold_snippets=["a passage", "another passage"])

    assert '"gold_snippets":["a passage","another passage"]' in question.model_dump_json()
