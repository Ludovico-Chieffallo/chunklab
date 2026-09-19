"""Statistical honesty: bootstrap confidence intervals over questions.

With a few dozen questions, a recall difference of a few points is often
noise. The paired bootstrap resamples *questions* (keeping each question's
pair of per-strategy scores together), which respects the fact that the same
questions are evaluated under every strategy.
"""

import numpy as np

#: A named winner must come out on top in at least this share of resamples.
#: Calibration was measured over 1,500 simulated comparisons (3-5 candidates,
#: correlated per-question outcomes, n in {18, 30, 60, 129}): picks reported in
#: the 90-100% band were correct 96% of the time, matching the 95% convention
#: used everywhere else here. Lower bands are materially less reliable - the
#: 80-90% band was right 82% of the time - which is why the bar is not lower.
SELECTION_CONFIDENCE = 0.90

#: Resampling in blocks keeps peak memory flat: a 889-question corpus under five
#: strategies would otherwise materialise a 355 MB intermediate.
_RESAMPLE_BLOCK_CELLS = 2_000_000


def probability_best(
    per_question: list[list[float]], resamples: int = 10_000, seed: int = 0
) -> list[float]:
    """For each candidate, the share of paired bootstrap resamples it wins.

    Comparing k candidates and then testing the top two is a *selection*, not a
    hypothesis test. The maximum of k noisy estimates is biased upward, so a
    pairwise interval computed after the field has been sorted overstates how
    sure the lead is — on a real 15-cell run the leading cell was genuinely
    best in only 68% of resamples while the report spoke as if the pair at the
    top were the only two candidates.

    Counting how often each candidate wins prices that selection in directly:
    the probabilities sum to one over the whole field, so there is nothing left
    to correct for. Classical corrections were tried first and rejected — Holm
    over leader-versus-rest erased a real and separately reproduced result on
    the example corpus, and a Hansen model confidence set retained every
    candidate on both corpora, which is less informative than saying nothing.

    Questions are resampled as a block, keeping each question's scores for
    every candidate together, exactly as `paired_bootstrap_diff_ci` does. Exact
    ties split the win equally.
    """
    if not per_question:
        return []
    if len({len(scores) for scores in per_question}) != 1:
        raise ValueError("probability_best requires equal-length score lists")

    arr = np.asarray(per_question, dtype=float)  # candidates x questions
    candidates, n = arr.shape
    if n == 0:
        return [0.0] * candidates
    if candidates == 1:
        return [1.0]

    rng = np.random.default_rng(seed)
    block = max(1, _RESAMPLE_BLOCK_CELLS // n)
    wins = np.zeros(candidates)
    drawn = 0
    while drawn < resamples:
        size = min(block, resamples - drawn)
        idx = rng.integers(0, n, size=(size, n))
        means = np.empty((candidates, size))
        for i in range(candidates):
            means[i] = arr[i][idx].mean(axis=1)
        # `>=` with a tolerance rather than argmax: two candidates that scored
        # identically on every question must split the win, not lose it to
        # whichever happens to sit earlier in the list.
        top = means >= means.max(axis=0) - 1e-12
        wins += (top / top.sum(axis=0)).sum(axis=1)
        drawn += size
    return (wins / resamples).tolist()


def minimal_confident_set(probabilities: list[float], level: float = 0.95) -> list[int]:
    """Indices of the fewest candidates whose win probabilities reach `level`.

    Returned best-first. The complement is what this corpus and question set
    can actually rule out, which is the decidable part of an undecided run.
    """
    order = sorted(range(len(probabilities)), key=lambda i: (-probabilities[i], i))
    chosen: list[int] = []
    total = 0.0
    for i in order:
        chosen.append(i)
        total += probabilities[i]
        if total >= level:
            break
    return chosen


def bootstrap_mean_ci(
    values: list[float], resamples: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of `values`."""
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(arr), size=(resamples, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def paired_bootstrap_diff_ci(
    a: list[float], b: list[float], resamples: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI for mean(a) - mean(b), resampling question pairs."""
    if len(a) != len(b):
        raise ValueError("paired bootstrap requires equal-length score lists")
    if not a:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    idx = rng.integers(0, len(diffs), size=(resamples, len(diffs)))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


#: Beyond this, "add more questions" stops being advice a human can act on. Nobody
#: hand-writes six figures of gold snippets, so a larger estimate is reported as
#: "no realistic number" instead of a number.
MAX_ACTIONABLE_QUESTIONS = 100_000


def estimate_questions_to_separate(
    n: int, observed_diff: float, ci: tuple[float, float]
) -> int | None:
    """Rough sample size at which the CI half-width would shrink below the
    observed difference (half-width scales ~ 1/sqrt(n)).

    Returns None when the answer would not be actionable: the difference is
    indistinguishable from zero, or the projected count exceeds
    `MAX_ACTIONABLE_QUESTIONS`. The estimate grows with 1/diff², so a difference
    of 1.8e-5 on 889 questions projected to 1.4 *billion* questions — a true
    number, and a useless one to print.
    """
    half_width = (ci[1] - ci[0]) / 2
    if abs(observed_diff) < 1e-9 or half_width <= 0:
        return None
    factor = half_width / abs(observed_diff)
    needed = max(n + 1, int(np.ceil(n * factor * factor)))
    return None if needed > MAX_ACTIONABLE_QUESTIONS else needed
