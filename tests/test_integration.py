"""End-to-end: sample doc + questions -> ranked report + all output formats."""

from pathlib import Path

import pytest

from chunklab.config import default_config
from chunklab.report.html import write_html_report
from chunklab.report.json_report import write_json_report
from chunklab.runner import evaluate

EXAMPLES = Path(__file__).parent.parent / "examples"
TEST_DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def report():
    config = default_config()
    config.embedding.backend = "fake"
    return evaluate(
        docs=TEST_DATA,
        questions=TEST_DATA / "questions.example.yaml",
        config=config,
    )


def test_report_structure(report):
    assert len(report.strategy_results) == 5
    assert report.recommendation
    # q11 has no gold snippets -> warning
    assert any("no gold snippets" in w for w in report.warnings)
    assert report.corpus_summary["num_scored_questions"] == 18


def test_ranked_best_first(report):
    metric = report.corpus_summary["ranking_metric"]
    attr = "balanced_score" if metric == "balanced" else metric
    values = [getattr(r, attr) for r in report.strategy_results]
    assert values == sorted(values, reverse=True)


def test_retrieval_finds_answers(report):
    best = report.strategy_results[0]
    assert best.hit_rate_at_k > 0.5


def test_outputs_written(report, tmp_path):
    html = write_html_report(report, tmp_path / "report.html")
    js = write_json_report(report, tmp_path / "report.json")
    html_text = html.read_text()
    assert "<title>ChunkLab report</title>" in html_text
    assert "Ranked comparison" in html_text
    import json

    data = json.loads(js.read_text())
    assert data["strategy_results"][0]["strategy"] == report.strategy_results[0].strategy


@pytest.mark.slow
def test_real_model_floor_beats_no_floor():
    """With real embeddings, floored semantic must beat semantic_no_floor."""
    config = default_config()
    report = evaluate(
        docs=TEST_DATA,
        questions=TEST_DATA / "questions.example.yaml",
        config=config,
    )
    by_name = {r.strategy: r for r in report.strategy_results}
    assert by_name["semantic"].recall_at_k >= by_name["semantic_no_floor"].recall_at_k
