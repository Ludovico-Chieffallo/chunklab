"""The HTML report must describe the run that happened, matrix runs included.

With `--compare-retrievers` it did not. Measured on a 5 x 3 run: the ranked table
had 15 rows carrying only the strategy name, each repeated three times, and the word
"dense" appeared nowhere in the file; the per-question drill-down was keyed by
strategy alone, so for two retrievers out of three it showed another retriever's
results; and the chunk diagnostics, identical under every retriever, came 15 times.
The console had been fixed for the matrix; the report people open next had not.
"""

import re

from chunklab.config import default_config
from chunklab.models import (
    ChunkHealth,
    Document,
    EvalReport,
    Question,
    QuestionResult,
    StrategyResult,
)
from chunklab.report.html import write_html_report
from chunklab.runner import run_evaluation


def _health() -> ChunkHealth:
    return ChunkHealth(
        num_chunks=4,
        tokens_min=10,
        tokens_median=20.0,
        tokens_mean=20.0,
        tokens_max=30,
        pct_tiny=0.0,
        pct_oversized=0.0,
        boundary_health=1.0,
        token_histogram=[(0, 4)],
    )


def _cell(strategy: str, retriever: str, hit: bool) -> StrategyResult:
    outcome = QuestionResult(
        question_id="q1",
        strategy=strategy,
        retrieved=[],
        hit=hit,
        first_hit_rank=1 if hit else None,
        gold_found_count=int(hit),
        gold_total=1,
    )
    return StrategyResult(
        strategy=strategy,
        retriever=retriever,
        recall_at_k=float(hit),
        hit_rate_at_k=float(hit),
        mrr=float(hit),
        precision_at_k=0.2 * hit,
        chunk_health=_health(),
        per_question=[outcome],
    )


def _report(cells: list[StrategyResult]) -> EvalReport:
    return EvalReport(
        corpus_summary={
            "num_documents": 1,
            "num_scored_questions": 1,
            "top_k": 5,
            "embedding_model": "test-model",
            "retrieval_modes": sorted({c.retriever for c in cells}),
            "queries": {"q1": "How long are records kept?"},
        },
        strategy_results=cells,
        recommendation="No single winner.",
    )


#: Each cell's own outcome on q1, chosen so that no two cells of a strategy agree.
MATRIX = [
    ("recursive", "dense", True),
    ("recursive", "bm25", False),
    ("fixed", "dense", False),
    ("fixed", "bm25", True),
]


def _render(report: EvalReport, tmp_path) -> str:
    return write_html_report(report, tmp_path / "report.html").read_text(encoding="utf-8")


def _section(html: str, heading: str) -> str:
    return html.split(f"<h2>{heading}</h2>")[1].split("<h2>")[0]


def _ranked_rows(html: str) -> list[str]:
    table = _section(html, "Ranked comparison")
    return re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.S)[1:]  # skip the header


def test_every_matrix_row_names_its_retriever(tmp_path):
    html = _render(_report([_cell(*cell) for cell in MATRIX]), tmp_path)

    rows = _ranked_rows(html)
    assert len(rows) == len(MATRIX)
    for row, (strategy, retriever, _hit) in zip(rows, MATRIX, strict=True):
        cells = [
            re.sub(r"<[^>]+>", "", cell).lstrip("✓✗ ").strip()
            for cell in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        ]
        assert cells[0] == strategy and cells[1] == retriever, row


def test_the_drill_down_shows_each_cells_own_result(tmp_path):
    html = _render(_report([_cell(*cell) for cell in MATRIX]), tmp_path)

    drill_down = _section(html, "Per-question drill-down")
    chips = re.findall(r'<span class="chip (hit|miss|split)">\s*(.*?)\s*</span>', drill_down, re.S)
    status = {" ".join(label.split()).split(" #")[0]: kind for kind, label in chips}
    assert status == {
        "recursive · dense": "hit",
        "recursive · bm25": "miss",
        "fixed · dense": "miss",
        "fixed · bm25": "hit",
    }


def test_chunk_diagnostics_appear_once_per_strategy(tmp_path):
    """They depend on the chunks alone, so every retriever reports the same ones."""
    html = _render(_report([_cell(*cell) for cell in MATRIX]), tmp_path)

    diagnostics = _section(html, "Per-strategy diagnostics")
    assert diagnostics.count("<details") == 2


def test_a_single_retriever_report_has_no_retriever_column(tmp_path):
    html = _render(
        _report([_cell("recursive", "dense", True), _cell("fixed", "dense", False)]), tmp_path
    )

    assert "<th>Retriever</th>" not in html
    drill_down = _section(html, "Per-question drill-down")
    assert "·" not in drill_down


def test_a_real_matrix_run_renders_all_three_retrievers(tmp_path):
    text = (
        "# Records\n\nRecords are kept for thirty days unless a legal hold applies.\n\n"
        "# Billing\n\nInvoices are payable within forty-five days of issue.\n"
    )
    document = Document(id="policy", source_path="policy.md", text=text)
    questions = [
        Question(
            id="q1",
            query="How long are records kept?",
            gold_snippets=["Records are kept for thirty days unless a legal hold applies"],
        )
    ]
    config = default_config()
    config.embedding.backend = "fake"
    config.retrieval.compare = ["dense", "bm25", "hybrid"]

    html = _render(run_evaluation([document], questions, config), tmp_path)

    rows = " ".join(_ranked_rows(html))
    for retriever in ("dense", "bm25", "hybrid"):
        assert f">{retriever}<" in rows
