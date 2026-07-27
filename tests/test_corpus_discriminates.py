"""Acceptance tests for the example corpus (roadmap phase 1.3).

Fast tests validate the corpus data itself with no model download. The `slow`
tests run the real embedding model and enforce the approved acceptance
criteria: aggregate recall swing >= 0.10, max per-document swing >= 0.15, and
at least two distinct per-document winners.
"""

from collections import defaultdict
from pathlib import Path

import pytest

from chunklab.config import load_questions
from chunklab.eval.gold_match import normalize
from chunklab.loaders.registry import load_documents

EXAMPLES = Path(__file__).parent.parent / "examples"
CORPUS = EXAMPLES / "corpus"
QUESTIONS = EXAMPLES / "questions.yaml"

EXPECTED_DOCS = {
    "faq_support", "contract_msa", "api_reference", "whitepaper", "policy_tables",
}


def _src_of(question) -> str:
    for tag in question.tags:
        if tag.startswith("src:"):
            return tag.removeprefix("src:")
    return "?"


# ---------- fast (no model) ----------


def test_corpus_loads_all_formats():
    docs = load_documents(CORPUS)
    assert {d.id for d in docs} == EXPECTED_DOCS
    formats = {d.metadata.get("format") for d in docs}
    assert {"md", "pdf", "docx"} <= formats


def test_questions_are_wellformed():
    questions = load_questions(QUESTIONS)
    scored = [q for q in questions if q.gold_snippets]
    assert len(scored) >= 40
    ids = [q.id for q in questions]
    assert len(ids) == len(set(ids))
    tags = {t for q in questions for t in q.tags}
    for required in ("needs_context", "table", "short_answer", "multi_snippet", "cross_doc",
                     "boundary", "two_sentence"):
        assert required in tags, f"tag '{required}' missing from the question set"


def test_every_gold_is_verbatim_in_corpus():
    docs = load_documents(CORPUS)
    corpus = " \n ".join(normalize(d.text) for d in docs)
    questions = load_questions(QUESTIONS)
    missing = [
        (q.id, gold[:50])
        for q in questions
        for gold in q.gold_snippets
        if normalize(gold) not in corpus
    ]
    assert not missing, f"gold snippets not found verbatim: {missing}"


# ---------- slow (real embedding model) ----------


@pytest.fixture(scope="module")
def corpus_report():
    from chunklab.runner import evaluate

    return evaluate(docs=CORPUS, questions=QUESTIONS)


@pytest.mark.slow
def test_corpus_separates_strategies(corpus_report):
    scores = [r.recall_at_k for r in corpus_report.strategy_results]
    assert max(scores) - min(scores) >= 0.10, (
        f"aggregate recall swing {max(scores) - min(scores):.3f} below 0.10"
    )


@pytest.mark.slow
def test_per_document_swing(corpus_report):
    questions = {q.id: q for q in load_questions(QUESTIONS)}
    per_doc_swings = {}
    per_doc = defaultdict(dict)
    for r in corpus_report.strategy_results:
        by_src = defaultdict(list)
        for qr in r.per_question:
            src = _src_of(questions[qr.question_id])
            if src in EXPECTED_DOCS:
                by_src[src].append(qr.gold_found_count / qr.gold_total)
        for src, vals in by_src.items():
            per_doc[src][r.strategy] = sum(vals) / len(vals)
    for src, by_strategy in per_doc.items():
        per_doc_swings[src] = max(by_strategy.values()) - min(by_strategy.values())
    assert max(per_doc_swings.values()) >= 0.15, (
        f"no document shows a >= 0.15 swing: {per_doc_swings}"
    )


@pytest.mark.slow
def test_no_universal_winner(corpus_report):
    questions = {q.id: q for q in load_questions(QUESTIONS)}
    per_doc = defaultdict(dict)
    for r in corpus_report.strategy_results:
        by_src = defaultdict(list)
        for qr in r.per_question:
            src = _src_of(questions[qr.question_id])
            if src in EXPECTED_DOCS:
                by_src[src].append(qr.gold_found_count / qr.gold_total)
        for src, vals in by_src.items():
            per_doc[src][r.strategy] = sum(vals) / len(vals)
    winners = {
        src: max(by_strategy, key=by_strategy.get) for src, by_strategy in per_doc.items()
    }
    assert len(set(winners.values())) >= 2, f"universal winner detected: {winners}"


@pytest.mark.slow
def test_balanced_changes_the_decision_on_real_data():
    """The context penalty is not cosmetic: on the API-reference document it
    reverses the top two strategies relative to raw recall.

    `fixed` retrieves more recall but pays ~2.2x the context cost of
    `structure`, so `balanced` prefers `structure`. This pins the claim in
    docs/metrics.md that the metric changes decisions on non-degenerate data.
    """
    from chunklab.config import default_config
    from chunklab.runner import run_evaluation

    docs = load_documents(CORPUS / "api_reference.md")
    questions = [q for q in load_questions(QUESTIONS) if "src:api_reference" in q.tags]

    report = run_evaluation(docs, questions, default_config())
    by = {r.strategy: r for r in report.strategy_results}

    assert by["fixed"].recall_at_k > by["structure"].recall_at_k
    assert by["structure"].balanced_score > by["fixed"].balanced_score
    assert report.strategy_results[0].strategy == "structure"
