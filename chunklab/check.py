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
"""

from dataclasses import dataclass, field
from pathlib import Path

from chunklab.eval.significance import paired_bootstrap_diff_ci
from chunklab.models import EvalReport, StrategyResult


@dataclass
class CheckOutcome:
    strategy: str
    retriever: str
    baseline_recall: float
    current_recall: float
    difference: float
    ci95: tuple[float, float] | None
    comparable: bool
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def _per_question_recalls(result: StrategyResult) -> dict[str, float]:
    return {
        q.question_id: q.gold_found_count / q.gold_total
        for q in result.per_question
        if q.gold_total > 0
    }


def load_baseline(path: str | Path) -> EvalReport:
    report = EvalReport.model_validate_json(Path(path).read_text(encoding="utf-8"))
    baseline_major = report.schema_version.split(".")[0]
    current_major = EvalReport().schema_version.split(".")[0]
    if baseline_major != current_major:
        raise ValueError(
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
        raise ValueError(f"no strategy matching '{wanted}' in the report")
    return candidates[0]


def compare(
    baseline: EvalReport,
    current: EvalReport,
    strategy: str | None = None,
    max_drop: float | None = None,
    resamples: int = 10_000,
    seed: int = 0,
) -> CheckOutcome:
    """Compare `current` against `baseline` on one strategy.

    With no `strategy`, the baseline's top-ranked entry is tracked — the one
    whose choice the baseline was recorded to defend.
    """
    tracked = _select(baseline, strategy, None)
    now = _select(current, tracked.strategy, tracked.retriever)

    base_scores = _per_question_recalls(tracked)
    now_scores = _per_question_recalls(now)
    shared = sorted(set(base_scores) & set(now_scores))

    outcome = CheckOutcome(
        strategy=tracked.strategy,
        retriever=tracked.retriever,
        baseline_recall=tracked.recall_at_k,
        current_recall=now.recall_at_k,
        difference=now.recall_at_k - tracked.recall_at_k,
        ci95=None,
        comparable=False,
    )

    base_hash = baseline.corpus_summary.get("questions_sha256")
    now_hash = current.corpus_summary.get("questions_sha256")
    if base_hash and now_hash and base_hash != now_hash:
        outcome.notes.append(
            "the question set changed since the baseline, so the two runs are not paired; "
            "only the raw difference is reported"
        )
    elif shared and len(shared) == len(base_scores) == len(now_scores):
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

    if not outcome.comparable and not outcome.failures:
        outcome.notes.append(
            "no statistical gate was applied; add --max-drop for a threshold check"
        )
    return outcome
