"""Picking the best of k candidates is a selection, not a pairwise test.

Sorting k noisy estimates and then testing the top two overstates the lead:
the maximum of k estimates is biased upward, and the interval is computed as
if those two had been the only candidates. On a real 15-cell run the leading
cell was genuinely best in 68% of resamples while the report spoke as though
the pair at the top were the whole field.

Two classical repairs were measured on this project's own corpora and rejected
before `probability_best` was written, which is why the tests below pin
behaviour rather than a formula:

* Holm over leader-versus-rest erased `recursive - semantic_no_floor` on the
  example corpus (+0.111, bootstrap p = 0.022) — a result reproduced
  independently and guarded by `tests/test_claims.py`.
* A Hansen model confidence set retained all five candidates on *both* the
  example corpus (129 questions) and the CRISPR run (18), which is less
  informative than the uncorrected report it was meant to fix.

Counting how often each candidate wins needs no correction — the
probabilities sum to one over the field — and it degrades into information
rather than into silence.
"""

import numpy as np
import pytest

from chunklab.eval.significance import (
    SELECTION_CONFIDENCE,
    minimal_confident_set,
    probability_best,
)


def test_a_dominant_candidate_takes_all_the_probability():
    chances = probability_best([[1.0] * 30, [0.0] * 30], resamples=2000, seed=0)
    assert chances[0] > 0.99
    assert sum(chances) == pytest.approx(1.0)


def test_identical_candidates_split_the_win():
    """Two strategies scoring the same on every question must share the credit,
    not hand it to whichever sits earlier in the list."""
    scores = [1.0, 0.0, 1.0, 0.5, 0.0, 1.0]
    chances = probability_best([scores, list(scores)], resamples=2000, seed=0)
    assert chances == pytest.approx([0.5, 0.5])


def test_probabilities_sum_to_one_over_the_field():
    """This is what makes a multiplicity correction unnecessary."""
    rng = np.random.default_rng(0)
    field = [list(rng.random(25)) for _ in range(6)]
    assert sum(probability_best(field, resamples=2000, seed=1)) == pytest.approx(1.0)


def test_more_evidence_raises_confidence_in_the_same_lead():
    """The gate must be able to open. A fixed true gap of 0.30 has to clear
    SELECTION_CONFIDENCE once there are enough questions to see it."""
    rng = np.random.default_rng(3)
    small = [list((rng.random(12) < p).astype(float)) for p in (0.65, 0.35)]
    large = [list((rng.random(600) < p).astype(float)) for p in (0.65, 0.35)]

    assert probability_best(large, resamples=2000, seed=0)[0] > SELECTION_CONFIDENCE
    assert probability_best(small, resamples=2000, seed=0)[0] > 0.5


def test_a_single_candidate_is_certain():
    assert probability_best([[0.4, 0.6]]) == [1.0]


def test_empty_and_unscored_fields_do_not_raise():
    assert probability_best([]) == []
    assert probability_best([[], []]) == [0.0, 0.0]


def test_mismatched_lengths_are_refused():
    """Unequal score lists mean the candidates were not scored on the same
    questions, and pairing them would be a lie."""
    with pytest.raises(ValueError, match="equal-length"):
        probability_best([[1.0, 0.0], [1.0]])


def test_blocked_resampling_matches_a_single_pass(monkeypatch):
    """Peak memory is capped by resampling in blocks; the result must not
    depend on where the block boundary falls."""
    import chunklab.eval.significance as sig

    field = [[1.0, 0.0, 1.0, 0.5] * 5, [0.0, 1.0, 0.5, 0.5] * 5]
    whole = probability_best(field, resamples=4000, seed=11)
    monkeypatch.setattr(sig, "_RESAMPLE_BLOCK_CELLS", 40)
    blocked = probability_best(field, resamples=4000, seed=11)

    assert blocked == pytest.approx(whole, abs=0.05)


# --- the minimal set is the decidable part of an undecided run ------------------


def test_minimal_set_keeps_the_fewest_candidates_reaching_the_level():
    assert minimal_confident_set([0.5, 0.45, 0.04, 0.01]) == [0, 1]


def test_minimal_set_excludes_nobody_when_the_field_is_flat():
    assert len(minimal_confident_set([0.25] * 4)) == 4


def test_minimal_set_is_a_single_candidate_when_one_dominates():
    assert minimal_confident_set([0.97, 0.02, 0.01]) == [0]


def test_minimal_set_breaks_ties_deterministically():
    assert minimal_confident_set([0.3, 0.3, 0.3, 0.1]) == minimal_confident_set(
        [0.3, 0.3, 0.3, 0.1]
    )
