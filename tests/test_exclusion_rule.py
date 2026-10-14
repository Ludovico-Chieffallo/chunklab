"""A candidate is ruled out only when the evidence is against it, not merely thin.

An undecided verdict names what can be ruled out: everything outside the fewest
candidates whose chances of being best add up to 95%. Alone, that rule ruled out a
strategy whose recall was never below the leader's by more than noise. On a 26-paper
corpus `recursive` was out with a 4% chance of being best while its paired 95% CI
against the leader was [+0.000, +0.111]; `semantic_no_floor`, with the same recall,
stayed in because it was more variable; and with `seed=3` `recursive` was back in,
because the cumulative 95% fell exactly on it. A verdict that turns on the seed is
not one to act on.

Now a candidate must also lose to the leader by more than sampling noise: the paired
95% CI of the difference must exclude zero.
"""

import re

from chunklab.config import default_config
from chunklab.eval.significance import (
    minimal_confident_set,
    probability_best,
    probability_not_worse,
)
from chunklab.models import ChunkHealth, QuestionResult, StrategyResult
from chunklab.runner import _build_recommendation

HEALTH = ChunkHealth(
    num_chunks=10,
    tokens_min=100,
    tokens_median=400,
    tokens_mean=400,
    tokens_max=800,
    pct_tiny=0.0,
    pct_oversized=0.0,
    boundary_health=1.0,
)


def _result(name: str, scores: list[float]) -> StrategyResult:
    per_question = [
        QuestionResult(
            question_id=f"q{i}",
            strategy=name,
            retrieved=[],
            hit=s > 0,
            gold_found_count=int(s),
            gold_total=1,
        )
        for i, s in enumerate(scores)
    ]
    mean = sum(scores) / len(scores)
    return StrategyResult(
        strategy=name,
        recall_at_k=mean,
        hit_rate_at_k=mean,
        mrr=mean,
        precision_at_k=mean,
        retrieved_tokens_at_k=2000.0,
        chunk_health=HEALTH,
        per_question=per_question,
    )


LEADER = [1.0] * 30 + [0.0] * 10
#: Trades two questions with the leader: as likely to be best.
RIVAL = [0.0, 0.0] + LEADER[2:30] + [1.0, 1.0] + LEADER[32:]
#: The leader minus two questions: rarely best, and never worse by more than noise.
CLOSE = LEADER[:2] + [0.0, 0.0] + LEADER[4:]
#: The leader minus ten questions: worse beyond noise.
FAR = LEADER[:4] + [0.0] * 10 + LEADER[14:]

RANKED = [
    _result("structure", LEADER),
    _result("semantic", RIVAL),
    _result("recursive", CLOSE),
    _result("fixed", FAR),
]


def _verdict(seed: int = 0) -> str:
    config = default_config()
    config.eval.seed = seed
    return _build_recommendation(RANKED, config, num_scored=len(LEADER))


def _decision(text: str) -> tuple[str, tuple[str, ...]]:
    """(the candidates still in play, the ones ruled out) - not the percentages,
    which move with the seed by a point and are meant to."""
    in_play = re.search(r"cannot be ruled out \(([^)]*)\)", text).group(1)
    out = re.findall(r"'(\w+)' \(\d+%\)", text.split("cannot be ruled out", 1)[1])
    return in_play, tuple(out)


def test_the_old_rule_alone_would_rule_out_the_close_candidate():
    chances = probability_best([LEADER, RIVAL, CLOSE, FAR], seed=0)

    assert 2 not in minimal_confident_set(chances)  # 'recursive', with ~3%


def test_a_candidate_within_noise_of_the_leader_is_not_ruled_out():
    text = _verdict()

    assert "3 cannot be ruled out ('structure', 'semantic', 'recursive')" in text, text
    assert "'fixed' (0%) can" in text, text


def test_the_verdict_does_not_turn_on_the_seed():
    assert len({_decision(_verdict(seed)) for seed in range(10)}) == 1


def test_probability_not_worse_at_the_extremes():
    scores = [1.0, 0.0, 1.0, 0.5]

    assert probability_not_worse(scores, scores, resamples=1000) == 1.0
    assert probability_not_worse([1.0] * 30, [0.0] * 30, resamples=1000) == 0.0


def test_a_borderline_exclusion_does_not_turn_on_the_seed():
    """Not worse than the leader in about 2.9% of resamples, just above the 2.5%
    line, as `semantic` was on the bundled example (2.76%). At 10,000 resamples
    that one flipped on one seed in fifty; the default must not."""
    leader = [1.0] * 10 + [0.0] * 3 + [0.5] * 116
    candidate = [0.0] * 10 + [1.0] * 3 + [0.5] * 116

    assert all(probability_not_worse(leader, candidate, seed=seed) >= 0.025 for seed in range(20))
