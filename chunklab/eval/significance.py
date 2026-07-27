"""Statistical honesty: bootstrap confidence intervals over questions.

With a few dozen questions, a recall difference of a few points is often
noise. The paired bootstrap resamples *questions* (keeping each question's
pair of per-strategy scores together), which respects the fact that the same
questions are evaluated under every strategy.
"""

import numpy as np


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
