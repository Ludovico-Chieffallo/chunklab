"""Console report: ranked table + recommendation (spec §5.4)."""

from rich.console import Console
from rich.table import Table

from chunklab.models import EvalReport


def print_report(report: EvalReport, console: Console | None = None) -> None:
    console = console or Console()
    cs = report.corpus_summary
    k = cs.get("top_k", 5)

    console.print(
        f"\n[bold]ChunkLab[/bold] — {cs.get('num_documents', '?')} document(s), "
        f"{cs.get('num_scored_questions', '?')} scored questions, top_k={k}, "
        f"model={cs.get('embedding_model', '?')}\n"
    )

    ranking = cs.get("ranking_metric", "recall_at_k")
    table = Table(show_edge=False)
    table.add_column("Strategy", style="bold")
    if ranking == "balanced":
        table.add_column("balanced", justify="right")
    table.add_column(f"recall@{k}", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column(f"prec@{k}", justify="right")
    table.add_column(f"tok@{k}", justify="right")
    table.add_column("#chunks", justify="right")
    table.add_column("med_tok", justify="right")
    table.add_column("%tiny", justify="right")
    table.add_column("boundary", justify="right")

    for i, r in enumerate(report.strategy_results):
        h = r.chunk_health
        style = "green" if i == 0 else None
        row = [("▶ " if i == 0 else "  ") + r.strategy]
        if ranking == "balanced":
            row.append(f"{r.balanced_score:.2f}")
        row += [
            f"{r.recall_at_k:.2f}",
            f"{r.mrr:.2f}",
            f"{r.precision_at_k:.2f}",
            f"{r.retrieved_tokens_at_k:.0f}",
            str(h.num_chunks),
            f"{h.tokens_median:.0f}",
            f"{h.pct_tiny:.0%}",
            f"{h.boundary_health:.0%}",
        ]
        table.add_row(*row, style=style)
    console.print(table)
    console.print(
        f"[dim]recall/MRR/prec: retrieval quality at k={k} · tok@{k}: mean tokens retrieved "
        "per question (context cost) · %tiny: chunks under the size floor · boundary: chunks "
        "not cut mid-sentence · full definitions: docs/metrics.md[/dim]"
    )

    console.print(f"\n[bold]Recommendation:[/bold]\n  {report.recommendation}")
    for w in report.warnings:
        console.print(f"[yellow]Warning:[/yellow] {w}")
