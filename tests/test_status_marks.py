"""The ranked table marks what the verdict decided, not whatever sorted first.

Its first row carried ▶, which reads as "the winner", on every run: on the bundled
example, whose verdict is "No single winner", and on a 26-paper matrix whose marked
cell held a strategy the text said could be ruled out.
"""

import re

from rich.console import Console

from chunklab.models import AxisVerdict, ChunkHealth, EvalReport, StrategyResult, Verdict
from chunklab.report.console import print_report
from chunklab.report.html import write_html_report

HEALTH = ChunkHealth(
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


def _row(strategy: str, recall: float, retriever: str = "dense") -> StrategyResult:
    return StrategyResult(
        strategy=strategy,
        retriever=retriever,
        recall_at_k=recall,
        hit_rate_at_k=recall,
        mrr=recall,
        precision_at_k=recall,
        chunk_health=HEALTH,
    )


def _axis(axis="chunking", decided=False, winner=None, leader="recursive", ruled_out=()):
    return AxisVerdict(
        axis=axis, decided=decided, winner=winner, leader=leader, ruled_out=list(ruled_out)
    )


def _report(rows, chunking, retrieval=None) -> EvalReport:
    return EvalReport(
        corpus_summary={"top_k": 5, "num_scored_questions": 40, "ranking_metric": "recall_at_k"},
        strategy_results=rows,
        verdict=Verdict(chunking=chunking, retrieval=retrieval),
    )


def _lines(report: EvalReport) -> dict[str, str]:
    console = Console(record=True, width=200, force_terminal=False)
    print_report(report, console)
    lines = console.export_text().splitlines()
    return {
        name: next(line for line in lines if re.search(rf"(^|\s){name}(\s|$)", line))
        for name in ("recursive", "structure", "semantic")
    }


ROWS = [_row("recursive", 0.82), _row("structure", 0.81), _row("semantic", 0.73)]


def test_an_undecided_run_marks_no_winner():
    lines = _lines(_report(ROWS, _axis(ruled_out=["semantic"])))

    assert "✓" not in lines["recursive"] and "▶" not in lines["recursive"]
    assert "✗" in lines["semantic"]
    assert "✗" not in lines["structure"]


def test_a_decided_run_marks_its_winner():
    lines = _lines(_report(ROWS, _axis(decided=True, winner="recursive")))

    assert "✓" in lines["recursive"]
    assert "✓" not in lines["structure"]


def test_a_matrix_row_is_out_when_its_retriever_is():
    rows = [_row("recursive", 0.8, "hybrid"), _row("recursive", 0.6, "dense")]
    report = _report(
        rows,
        _axis(decided=True, winner="recursive"),
        _axis(axis="retrieval", leader="hybrid", ruled_out=["dense"]),
    )
    console = Console(record=True, width=200, force_terminal=False)
    print_report(report, console)
    lines = [line for line in console.export_text().splitlines() if "recursive" in line]

    hybrid = next(line for line in lines if "hybrid" in line)
    dense = next(line for line in lines if "dense" in line)
    assert "✓" not in hybrid, "the retrieval axis is undecided, so no combination has won"
    assert "✗" in dense


def test_the_html_report_marks_rows_the_same_way(tmp_path):
    path = write_html_report(_report(ROWS, _axis(ruled_out=["semantic"])), tmp_path / "r.html")
    html = path.read_text(encoding="utf-8")

    assert "▶" not in html
    assert 'class="winner"' not in html
    assert "✗ semantic" in html
