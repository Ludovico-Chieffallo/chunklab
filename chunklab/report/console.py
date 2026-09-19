"""Console report: ranked table + recommendation (spec §5.4)."""

from rich.console import Console
from rich.table import Table

from chunklab.models import ChunkHealth, EvalReport

#: Diagnostics derived from the chunks alone. Every retriever a strategy is
#: evaluated under reports the same four numbers, so in a matrix they are the
#: same value repeated once per cell.
_CHUNK_HEALTH_COLUMNS = ("#chunks", "med_tok", "%tiny", "boundary")


def _chunk_health_cells(health: ChunkHealth) -> list[str]:
    return [
        str(health.num_chunks),
        f"{health.tokens_median:.0f}",
        f"{health.pct_tiny:.0%}",
        f"{health.boundary_health:.0%}",
    ]


def _print_chunk_health(report: EvalReport, console: Console) -> None:
    """Chunk diagnostics, once per strategy rather than once per matrix cell.

    Carrying them inline cost four columns on a table that already had a
    retriever column, and rich spends what is left truncating the first one: on
    a real 5 x 3 run at 88 columns every name collapsed to 'sem…', so the output
    could not distinguish 'semantic' from 'semantic_no_floor' — the two rows the
    whole fragment-trap argument rests on.
    """
    by_strategy: dict[str, ChunkHealth] = {}
    for result in report.strategy_results:
        by_strategy.setdefault(result.strategy, result.chunk_health)

    table = Table(show_edge=False)
    table.add_column("Strategy", style="bold")
    for name in _CHUNK_HEALTH_COLUMNS:
        table.add_column(name, justify="right")
    for strategy, health in by_strategy.items():
        table.add_row(f"  {strategy}", *_chunk_health_cells(health))

    console.print(
        "\n[bold]Chunk health[/bold] [dim]— per strategy; identical under every retriever[/dim]"
    )
    console.print(table)


def print_report(report: EvalReport, console: Console | None = None) -> None:
    console = console or Console()
    cs = report.corpus_summary
    k = cs.get("top_k", 5)

    # A scanned PDF loads and contributes nothing, so the headline count has to
    # say how many documents the index was actually built from.
    loaded = cs.get("num_documents", "?")
    with_text = cs.get("num_documents_with_text")
    documents = (
        f"{loaded} document(s)"
        if with_text is None or with_text == loaded
        else f"{loaded} document(s) ({with_text} with extractable text)"
    )
    console.print(
        f"\n[bold]ChunkLab[/bold] — {documents}, "
        f"{cs.get('num_scored_questions', '?')} scored questions, top_k={k}, "
        f"model={cs.get('embedding_model', '?')}\n"
    )

    ranking = cs.get("ranking_metric", "recall_at_k")
    # Only worth a column when there is something to compare against.
    show_retriever = len({r.retriever for r in report.strategy_results}) > 1

    table = Table(show_edge=False)
    table.add_column("Strategy", style="bold")
    if show_retriever:
        table.add_column("retriever")
    if ranking == "balanced":
        table.add_column("balanced", justify="right")
    table.add_column(f"recall@{k}", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column(f"prec@{k}", justify="right")
    table.add_column(f"tok@{k}", justify="right")
    if not show_retriever:
        for name in _CHUNK_HEALTH_COLUMNS:
            table.add_column(name, justify="right")

    for i, r in enumerate(report.strategy_results):
        style = "green" if i == 0 else None
        row = [("▶ " if i == 0 else "  ") + r.strategy]
        if show_retriever:
            row.append(r.retriever)
        if ranking == "balanced":
            row.append(f"{r.balanced_score:.2f}")
        row += [
            f"{r.recall_at_k:.2f}",
            f"{r.mrr:.2f}",
            f"{r.precision_at_k:.2f}",
            f"{r.retrieved_tokens_at_k:.0f}",
        ]
        if not show_retriever:
            row += _chunk_health_cells(r.chunk_health)
        table.add_row(*row, style=style)
    console.print(table)

    if show_retriever:
        _print_chunk_health(report, console)

    console.print(
        f"[dim]recall/MRR/prec: retrieval quality at k={k} · tok@{k}: mean tokens retrieved "
        "per question (context cost) · %tiny: chunks under the size floor · boundary: chunks "
        "not cut mid-sentence · full definitions: docs/metrics.md[/dim]"
    )
    # Measured, not hedged: swapping the model changed the winner on one of the two
    # corpora tested. Ranking a strategy without naming the model is meaningless.
    console.print(
        f"[dim]This ranking holds for {cs.get('embedding_model', 'this model')}. "
        "Strategy order changes with the embedding model — run with the one you deploy.[/dim]"
    )

    # A matrix report carries one block per axis, separated by a blank line; indent
    # each of them so the second does not hang off the left margin.
    body = "\n".join(f"  {line}" if line else "" for line in report.recommendation.split("\n"))
    console.print(f"\n[bold]Recommendation:[/bold]\n{body}")
    for w in report.warnings:
        console.print(f"[yellow]Warning:[/yellow] {w}")
