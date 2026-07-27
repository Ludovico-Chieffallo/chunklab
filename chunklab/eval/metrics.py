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


def retrieved_tokens_at_k(results: list[QuestionResult]) -> float:
    """Mean total tokens retrieved per question - the context cost paid downstream."""
    if not results:
        return 0.0
    per_q = [sum(rc.chunk.token_count for rc in r.retrieved) for r in results]
    return sum(per_q) / len(per_q)


def context_efficiency(results: list[QuestionResult], gold_tokens: dict[str, list[int]]) -> float:
    """Mean fraction of retrieved tokens that belong to a found gold snippet.

    `gold_tokens` maps question_id -> token count of each of its gold snippets.
    """
    if not results:
        return 0.0
    per_q = []
    for r in results:
        retrieved = sum(rc.chunk.token_count for rc in r.retrieved)
        if retrieved == 0:
            per_q.append(0.0)
            continue
        found = sum(gold_tokens[r.question_id][i] for i in r.found_gold_indices)
        per_q.append(min(found / retrieved, 1.0))
    return sum(per_q) / len(per_q)


def balanced_scores(
    recalls: dict[str, float], tokens: dict[str, float], lambda_: float
) -> dict[str, float]:
    """recall penalized by context cost relative to the leanest strategy.

    balanced(s) = recall(s) - lambda * (tokens(s) / min_tokens - 1). A strategy
    retrieving the fewest tokens pays no penalty; one retrieving twice the
    minimum pays `lambda`. Rationale and calibration: docs/metrics.md.
    """
    positive = [t for t in tokens.values() if t > 0]
    if not positive:
        return dict(recalls)
    t_min = min(positive)
    return {s: recalls[s] - lambda_ * (max(tokens[s], t_min) / t_min - 1.0) for s in recalls}
