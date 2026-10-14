"""Regression gate for CI: has retrieval quality dropped since the baseline?

Chunking is chosen once and then quietly rots: documents get rewritten, a loader
is upgraded, someone changes `chunk_size` to fix an unrelated problem. `chunklab
check` re-runs the pinned configuration and compares it against a stored report.

What makes it fail is the design decision that matters. The default gate is the
**paired bootstrap over questions**, the same test the recommendation uses: a
build fails when the drop is larger than sampling noise can explain, not when a
number moved. A hard threshold is available with `--max-drop`, off by default,
for teams that want a blunt floor as well - but a tool that refuses to name a
winner on noise must not fail a build on noise either.

Pairing requires the same questions on both sides; the report records
`questions_sha256` precisely so that can be checked rather than assumed.

When the two runs cannot be compared, the gate fails closed. It used to print
"No regression detected." and exit 0 when the questions had changed, which left
CI ungated, silently, until someone re-recorded the baseline. What makes two
runs incomparable is a change to the measurement itself: the questions, `top_k`,
the fuzzy threshold, or the chunklab version that scored them. A change to what
is measured - the corpus, the model, a strategy's parameters - is exactly what
the gate exists to catch, so it is compared and named in a note.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from chunklab.errors import InputError, InputNotFoundError, describe_validation_error
from chunklab.eval.significance import paired_bootstrap_diff_ci
from chunklab.models import EvalReport, StrategyResult


@dataclass
class CheckOutcome:
    strategy: str
    retriever: str
    baseline_recall: float
    current_recall: float | None  # None when the tracked entry is missing from the run
    difference: float | None
    ci95: tuple[float, float] | None
    comparable: bool
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: Why the two runs cannot be compared; empty when they can.
    not_comparable: list[str] = field(default_factory=list)
    allow_unpaired: bool = False

    @property
    def status(self) -> Literal["pass", "regression", "not_comparable"]:
        if self.not_comparable and not self.allow_unpaired:
            return "not_comparable"
        return "regression" if self.failures else "pass"

    @property
    def ok(self) -> bool:
        return self.status == "pass"


def _per_question_recalls(result: StrategyResult) -> dict[str, float]:
    return {
        q.question_id: q.gold_found_count / q.gold_total
        for q in result.per_question
        if q.gold_total > 0
    }


def load_baseline(path: str | Path) -> EvalReport:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise InputNotFoundError(
            f"baseline not found: {path}; record one with --update-baseline"
        ) from None
    try:
        report = EvalReport.model_validate_json(text)
    except ValidationError as exc:
        raise InputError(
            f"{path} is not a chunklab report: {describe_validation_error(exc)}"
        ) from None
    if not report.strategy_results:
        # Every field of EvalReport has a default, so any JSON object parses.
        raise InputError(f"{path} is not a chunklab report: it holds no strategy results")
    baseline_major = report.schema_version.split(".")[0]
    current_major = EvalReport().schema_version.split(".")[0]
    if baseline_major != current_major:
        raise InputError(
            f"baseline uses schema {report.schema_version}, this chunklab writes "
            f"{EvalReport().schema_version}; regenerate the baseline with --update-baseline"
        )
    return report


def _select(report: EvalReport, strategy: str | None, retriever: str | None) -> StrategyResult:
    candidates = report.strategy_results
    if strategy:
        candidates = [r for r in candidates if r.strategy == strategy]
    if retriever:
        candidates = [r for r in candidates if r.retriever == retriever]
    if not candidates:
        wanted = f"{strategy or 'any'} + {retriever or 'any'}"
        raise InputError(f"no strategy matching '{wanted}' in the report")
    return candidates[0]


def _measurement_changes(baseline: EvalReport, current: EvalReport) -> list[str]:
    """What changed in how the two runs were measured, which makes them incomparable."""
    before, now = baseline.corpus_summary, current.corpus_summary
    reasons = []
    if before.get("chunklab_version") != now.get("chunklab_version"):
        reasons.append(
            f"the baseline was scored by chunklab {before.get('chunklab_version') or '(unknown)'}"
            f" and this run by chunklab {now.get('chunklab_version') or '(unknown)'}; a new "
            "version can score the same retrieval differently (see the CHANGELOG)"
        )
    if before.get("top_k") != now.get("top_k"):
        reasons.append(
            f"the baseline retrieved the top {before.get('top_k')} chunks and this run the top "
            f"{now.get('top_k')}; recall at another k is another measure"
        )
    if before.get("fuzzy_threshold") != now.get("fuzzy_threshold"):
        reasons.append(
            f"the fuzzy threshold was {before.get('fuzzy_threshold')} for the baseline and is "
            f"{now.get('fuzzy_threshold')} now, so one chunk can count as found in one run and "
            "not in the other"
        )
    return reasons


def _system_changes(
    baseline: EvalReport, current: EvalReport, tracked: StrategyResult, now: StrategyResult
) -> list[str]:
    """What changed in the system under test: gated as usual, and named."""
    before, after = baseline.corpus_summary, current.corpus_summary
    notes = []
    if before.get("corpus_sha256") != after.get("corpus_sha256"):
        notes.append("the corpus changed since the baseline")
    for key, label in (
        ("embedding_model", "the embedding model"),
        ("embedding_model_revision", "the embedding model's revision"),
        ("embedding_prefixes", "the embedding prefixes setting"),
    ):
        if before.get(key) != after.get(key):
            notes.append(f"{label} changed: {before.get(key)} -> {after.get(key)}")
    if tracked.config != now.config:
        notes.append(
            f"'{tracked.strategy} + {tracked.retriever}' ran with {tracked.config} for the "
            f"baseline and {now.config} now"
        )
    return notes


def compare(
    baseline: EvalReport,
    current: EvalReport,
    strategy: str | None = None,
    max_drop: float | None = None,
    resamples: int = 10_000,
    seed: int = 0,
    allow_unpaired: bool = False,
) -> CheckOutcome:
    """Compare `current` against `baseline` on one strategy.

    With no `strategy`, the baseline's top-ranked entry is tracked — the one
    whose choice the baseline was recorded to defend. When the runs cannot be
    compared the status is `not_comparable`, unless `allow_unpaired`, which
    falls back to the raw difference and `max_drop`.
    """
    tracked = _select(baseline, strategy, None)
    matches = [
        r
        for r in current.strategy_results
        if (r.strategy, r.retriever) == (tracked.strategy, tracked.retriever)
    ]
    outcome = CheckOutcome(
        strategy=tracked.strategy,
        retriever=tracked.retriever,
        baseline_recall=tracked.recall_at_k,
        current_recall=None,
        difference=None,
        ci95=None,
        comparable=False,
        allow_unpaired=allow_unpaired,
        not_comparable=_measurement_changes(baseline, current),
    )
    if not matches:
        present = ", ".join(f"{r.strategy} + {r.retriever}" for r in current.strategy_results)
        outcome.not_comparable.append(
            f"the baseline's '{tracked.strategy} + {tracked.retriever}' is not in this run, "
            f"which evaluated {present or 'nothing'}"
        )
        return outcome
    now = matches[0]
    outcome.current_recall = now.recall_at_k
    outcome.difference = now.recall_at_k - tracked.recall_at_k
    outcome.notes.extend(_system_changes(baseline, current, tracked, now))

    base_scores = _per_question_recalls(tracked)
    now_scores = _per_question_recalls(now)
    shared = sorted(set(base_scores) & set(now_scores))

    base_hash = baseline.corpus_summary.get("questions_sha256")
    now_hash = current.corpus_summary.get("questions_sha256")
    if base_hash and now_hash and base_hash != now_hash:
        outcome.not_comparable.append(
            "the question set changed since the baseline, so the two runs cannot be paired "
            "question by question"
        )
    elif not (shared and len(shared) == len(base_scores) == len(now_scores)):
        outcome.not_comparable.append(
            f"the two runs scored different questions ({len(base_scores)} for the baseline, "
            f"{len(now_scores)} now, {len(shared)} in common), so they cannot be paired"
        )
    elif not outcome.not_comparable:
        outcome.comparable = True
        a = [now_scores[q] for q in shared]
        b = [base_scores[q] for q in shared]
        outcome.ci95 = paired_bootstrap_diff_ci(a, b, resamples=resamples, seed=seed)

    if outcome.comparable and outcome.ci95 is not None and outcome.ci95[1] < 0:
        outcome.failures.append(
            f"recall dropped {outcome.difference:+.3f} on '{outcome.strategy}"
            f" + {outcome.retriever}', 95% CI [{outcome.ci95[0]:+.3f}, {outcome.ci95[1]:+.3f}]"
            " entirely below zero - larger than sampling noise explains"
        )
    if max_drop is not None and outcome.difference < -abs(max_drop):
        outcome.failures.append(
            f"recall dropped {outcome.difference:+.3f}, beyond the --max-drop limit"
            f" of {abs(max_drop):.3f}"
        )

    if not outcome.comparable and allow_unpaired and not outcome.failures:
        outcome.notes.append(
            "no statistical gate was applied; add --max-drop for a threshold check"
        )
    return outcome
