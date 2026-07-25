"""Orchestrates the full evaluation pipeline (spec §3.3)."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from chunklab import __version__
from chunklab.chunkers.registry import make_chunker
from chunklab.config import Config, default_config, load_questions
from chunklab.diagnostics.chunk_health import compute_chunk_health
from chunklab.embeddings.registry import make_embedder
from chunklab.eval import metrics as m
from chunklab.eval.gold_match import score_question
from chunklab.eval.significance import bootstrap_mean_ci
from chunklab.loaders.registry import load_documents
from chunklab.models import Document, EvalReport, Question, StrategyResult
from chunklab.retrieval.dense import DenseRetriever
from chunklab.text_utils import count_tokens


def _corpus_sha256(documents: list[Document]) -> str:
    """SHA-256 over (doc_id, text) pairs sorted by doc_id — order-independent."""
    h = hashlib.sha256()
    for doc in sorted(documents, key=lambda d: d.id):
        h.update(doc.id.encode("utf-8"))
        h.update(b"\x00")
        h.update(doc.text.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _questions_sha256(questions: list[Question]) -> str:
    """SHA-256 over the canonical JSON of each question, sorted by id."""
    h = hashlib.sha256()
    for q in sorted(questions, key=lambda q: q.id):
        h.update(q.model_dump_json().encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _rank_key(result: StrategyResult, ranking_metric: str):
    attr = "balanced_score" if ranking_metric == "balanced" else ranking_metric
    primary = getattr(result, attr)
    return (
        -primary,
        -result.mrr,
        -result.hit_rate_at_k,
        result.chunk_health.pct_tiny,
    )


def _per_question_recalls(result: StrategyResult) -> list[float]:
    return [q.gold_found_count / q.gold_total for q in result.per_question if q.gold_total > 0]


def _build_recommendation(ranked: list[StrategyResult], config: Config, num_scored: int) -> str:
    if not ranked:
        return "No strategies were evaluated."
    best, worst = ranked[0], ranked[-1]
    metric = config.eval.ranking_metric
    metric_attr = "balanced_score" if metric == "balanced" else metric
    k = config.retrieval.top_k
    metric_label = metric.replace("_at_k", f"@{k}").replace("_", " ")
    params = ", ".join(f"{key}={val}" for key, val in best.config.items())

    # Statistical gate: recommend only if the top-2 gap survives a paired bootstrap.
    if len(ranked) > 1:
        from chunklab.eval.significance import (
            estimate_questions_to_separate,
            paired_bootstrap_diff_ci,
        )

        a, b = _per_question_recalls(ranked[0]), _per_question_recalls(ranked[1])
        if a and len(a) == len(b):
            diff = sum(a) / len(a) - sum(b) / len(b)
            ci = paired_bootstrap_diff_ci(
                a, b, resamples=config.eval.bootstrap_resamples, seed=config.eval.seed
            )
            if ci[0] <= 0.0 <= ci[1]:
                needed = estimate_questions_to_separate(len(a), diff, ci)
                needed_txt = (
                    f" Roughly {needed} scored questions would be needed to separate them"
                    " at the observed difference."
                    if needed
                    else ""
                )
                return (
                    f"No winner: '{ranked[0].strategy}' and '{ranked[1].strategy}' are"
                    f" statistically indistinguishable on {num_scored} scored questions"
                    f" (recall difference {diff:+.3f}, 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]"
                    f" includes zero).{needed_txt}"
                    " Add questions before committing to a strategy."
                )

    lines = [
        f"Use {best.strategy.upper()} chunking"
        + (f" ({params})." if params else ".")
        + f" It gave the best retrieval on your corpus "
        f"({metric_label} = {getattr(best, metric_attr):.2f})."
    ]

    if worst.chunk_health.pct_tiny >= 0.30 and worst.strategy != best.strategy:
        line = (
            f"{worst.strategy} scored worst ({metric_label} = "
            f"{getattr(worst, metric_attr):.2f}) "
            f"with {worst.chunk_health.pct_tiny:.0%} of its chunks under "
            f"{config.eval.min_floor_tokens} tokens (the fragment trap)."
        )
        floored = next((r for r in ranked if r.strategy == "semantic"), None)
        if worst.strategy == "semantic_no_floor" and floored:
            delta = getattr(floored, metric_attr) - getattr(worst, metric_attr)
            if delta > 0:
                line += f" The floored 'semantic' variant recovered {delta * 100:.0f} points."
        lines.append(line)

    for r in ranked:
        n_split = sum(1 for q in r.per_question if q.split_across_chunks)
        if n_split > 0:
            lines.append(
                f"{n_split} of {num_scored} questions had the answer split across two "
                f"chunks under '{r.strategy}'; increasing overlap or using "
                f"structure-aware chunking fixes this."
            )
            break

    if best.chunk_health.pct_oversized > 0:
        lines.append(
            f"Note: {best.chunk_health.pct_oversized:.0%} of {best.strategy} chunks exceed "
            f"the embedding model's max sequence length and may be truncated at embed time."
        )

    return " ".join(lines)


def run_evaluation(
    documents: list[Document], questions: list[Question], config: Config
) -> EvalReport:
    warnings: list[str] = []

    scored_questions = [q for q in questions if q.gold_snippets]
    skipped = len(questions) - len(scored_questions)
    if skipped:
        warnings.append(
            f"{skipped} question(s) had no gold snippets and were skipped; "
            "add gold_snippets to include them in scoring."
        )
    if not scored_questions:
        raise ValueError("no questions with gold snippets - nothing to score")

    if len(scored_questions) < 15:
        warnings.append(
            f"only {len(scored_questions)} scored questions: differences between strategies "
            "are unlikely to be statistically meaningful; aim for at least 15-20."
        )

    embedder = make_embedder(config.embedding.backend, config.embedding.model)
    doc_map = {d.id: d for d in documents}
    k = config.retrieval.top_k
    gold_tokens = {
        q.id: [count_tokens(g) for g in q.gold_snippets] for q in scored_questions
    }

    results: list[StrategyResult] = []
    chunks_by_strategy: dict[str, list] = {}
    for strategy in config.strategies:
        chunker = make_chunker(strategy.name, strategy.params, embedder=embedder)
        chunks = [c for doc in documents for c in chunker.chunk(doc)]
        if not chunks:
            warnings.append(f"strategy '{strategy.name}' produced no chunks; skipped.")
            continue
        chunks_by_strategy[strategy.name] = chunks

        retriever = DenseRetriever(chunks, embedder)
        per_question = [
            score_question(
                q,
                retriever.retrieve(q.query, k),
                strategy.name,
                config.eval.fuzzy_threshold,
            )
            for q in scored_questions
        ]

        health = compute_chunk_health(
            chunks,
            doc_map,
            min_floor_tokens=config.eval.min_floor_tokens,
            max_tokens=int(strategy.params.get("max_tokens", 10_000)),
            embedder_max_seq=embedder.max_seq_tokens,
        )
        results.append(
            StrategyResult(
                strategy=strategy.name,
                config=strategy.params,
                recall_at_k=m.recall_at_k(per_question),
                hit_rate_at_k=m.hit_rate_at_k(per_question),
                mrr=m.mrr(per_question),
                precision_at_k=m.precision_at_k(per_question, k),
                retrieved_tokens_at_k=m.retrieved_tokens_at_k(per_question),
                context_efficiency=m.context_efficiency(per_question, gold_tokens),
                chunk_health=health,
                per_question=per_question,
            )
        )

    # Balanced score needs every strategy's token cost (normalized on the minimum).
    balanced = m.balanced_scores(
        {r.strategy: r.recall_at_k for r in results},
        {r.strategy: r.retrieved_tokens_at_k for r in results},
        config.eval.balanced_lambda,
    )
    for r in results:
        r.balanced_score = balanced[r.strategy]
        r.ci95 = bootstrap_mean_ci(
            _per_question_recalls(r),
            resamples=config.eval.bootstrap_resamples,
            seed=config.eval.seed,
        )

    results.sort(key=lambda r: _rank_key(r, config.eval.ranking_metric))

    viz = None
    if documents and chunks_by_strategy:
        from chunklab.report.viz import build_doc_viz

        viz = build_doc_viz(documents[0], scored_questions, chunks_by_strategy).model_dump()

    return EvalReport(
        corpus_summary={
            "num_documents": len(documents),
            "documents": [d.id for d in documents],
            "num_questions": len(questions),
            "num_scored_questions": len(scored_questions),
            "embedding_model": config.embedding.model,
            "embedding_model_revision": embedder.revision,
            "top_k": k,
            "ranking_metric": config.eval.ranking_metric,
            "seed": config.eval.seed,
            "balanced_lambda": config.eval.balanced_lambda,
            "queries": {q.id: q.query for q in scored_questions},
            "chunklab_version": __version__,
            "corpus_sha256": _corpus_sha256(documents),
            "questions_sha256": _questions_sha256(scored_questions),
        },
        strategy_results=results,
        recommendation=_build_recommendation(results, config, len(scored_questions)),
        warnings=warnings,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        viz=viz,
    )


def evaluate(
    docs: str | Path | list[Document],
    questions: str | Path | list[Question],
    config: Config | str | Path | None = None,
) -> EvalReport:
    """Public API: evaluate chunking strategies over documents and questions.

    `docs` is a path (file or directory) or a list of Documents; `questions` is
    a YAML path or a list of Questions; `config` is a Config, a YAML path, or
    None for defaults.
    """
    if not isinstance(docs, list):
        docs = load_documents(docs)
    if not isinstance(questions, list):
        questions = load_questions(questions)
    if config is None:
        config = default_config()
    elif not isinstance(config, Config):
        from chunklab.config import load_config

        config = load_config(config)
    return run_evaluation(docs, questions, config)
