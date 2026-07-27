"""Run chunklab over a prepared benchmark and report aggregate results.

Handles both layouts produced by the prepare scripts:

* a single corpus (`<dir>/corpus`, `<dir>/questions.yaml`) — QASPER;
* many single-document corpora (`<dir>/<case>/corpus`, ...) — CUAD.

Aggregation pools **per-question** recall across cases rather than averaging
per-case means, so a contract with 30 annotated categories weighs more than one
with 3 — each question is one observation, which is also what makes the paired
bootstrap over the pooled values legitimate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from chunklab.config import default_config, load_questions  # noqa: E402
from chunklab.eval.significance import (  # noqa: E402
    estimate_questions_to_separate,
    paired_bootstrap_diff_ci,
)
from chunklab.loaders.registry import load_documents  # noqa: E402
from chunklab.runner import run_evaluation  # noqa: E402


def discover_cases(root: Path) -> list[Path]:
    if (root / "questions.yaml").exists():
        return [root]
    return sorted(p for p in root.iterdir() if (p / "questions.yaml").exists())


def evaluate_case(case: Path, config) -> dict[str, dict]:
    documents = load_documents(case / "corpus")
    questions = load_questions(case / "questions.yaml")
    report = run_evaluation(documents, questions, config)
    return {
        result.strategy: {
            "recalls": [
                q.gold_found_count / q.gold_total for q in result.per_question if q.gold_total
            ],
            "tokens": result.retrieved_tokens_at_k,
            "mrr": result.mrr,
            "questions": len(result.per_question),
        }
        for result in report.strategy_results
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, required=True, help="Prepared benchmark directory.")
    parser.add_argument("--label", default="benchmark")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    config = default_config()
    config.retrieval.top_k = args.top_k

    cases = discover_cases(args.bench)
    pooled: dict[str, list[float]] = {}
    token_cost: dict[str, list[tuple[float, int]]] = {}
    mrr_cost: dict[str, list[tuple[float, int]]] = {}

    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case.name}", flush=True)
        for strategy, data in evaluate_case(case, config).items():
            pooled.setdefault(strategy, []).extend(data["recalls"])
            token_cost.setdefault(strategy, []).append((data["tokens"], data["questions"]))
            mrr_cost.setdefault(strategy, []).append((data["mrr"], data["questions"]))

    def weighted(pairs: list[tuple[float, int]]) -> float:
        total = sum(n for _, n in pairs)
        return sum(value * n for value, n in pairs) / max(total, 1)

    rows = sorted(
        (
            (
                strategy,
                sum(recalls) / len(recalls),
                weighted(mrr_cost[strategy]),
                weighted(token_cost[strategy]),
            )
            for strategy, recalls in pooled.items()
        ),
        key=lambda row: -row[1],
    )

    n_questions = len(next(iter(pooled.values())))
    print(f"\n=== {args.label}: {len(cases)} case(s), {n_questions} questions, k={args.top_k} ===")
    print(f"{'strategy':20s} {'recall@k':>9s} {'MRR':>7s} {'tok@k':>8s}")
    for strategy, recall, mrr, tokens in rows:
        print(f"{strategy:20s} {recall:9.3f} {mrr:7.3f} {tokens:8.0f}")

    best, second = rows[0][0], rows[1][0]
    a, b = pooled[best], pooled[second]
    diff = sum(a) / len(a) - sum(b) / len(b)
    low, high = paired_bootstrap_diff_ci(a, b, resamples=10_000, seed=0)
    separated = not (low <= 0.0 <= high)
    print(
        f"\ntop-2 paired bootstrap: {best} - {second} = {diff:+.3f},"
        f" 95% CI [{low:+.3f}, {high:+.3f}]"
    )
    if separated:
        print(f"=> '{best}' is significantly better on {n_questions} questions.")
    else:
        needed = estimate_questions_to_separate(len(a), diff, (low, high))
        if needed:
            print(f"=> statistically indistinguishable; ~{needed} questions would be needed.")
        else:
            cheapest = min(rows[:2], key=lambda row: row[3])
            print(
                "=> statistically indistinguishable, and no realistic number of questions"
                f" would separate them. Cheaper on context: '{cheapest[0]}'"
                f" ({cheapest[3]:.0f} tokens/question)."
            )


if __name__ == "__main__":
    main()
