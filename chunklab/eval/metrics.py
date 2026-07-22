"""Retrieval metrics over scored questions (spec §4.5)."""

from chunklab.models import QuestionResult


def hit_rate_at_k(results: list[QuestionResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.hit) / len(results)


def recall_at_k(results: list[QuestionResult]) -> float:
    if not results:
        return 0.0
    per_q = [r.gold_found_count / r.gold_total for r in results if r.gold_total > 0]
    return sum(per_q) / len(per_q) if per_q else 0.0


def mrr(results: list[QuestionResult]) -> float:
    if not results:
        return 0.0
    return sum(1.0 / r.first_hit_rank for r in results if r.first_hit_rank) / len(results)


def precision_at_k(results: list[QuestionResult], k: int) -> float:
    if not results or k <= 0:
        return 0.0
    per_q = [sum(1 for rc in r.retrieved if rc.is_hit) / k for r in results]
    return sum(per_q) / len(per_q)
