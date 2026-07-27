"""Typer CLI for chunklab."""

from pathlib import Path

import typer
from rich.console import Console

import chunklab

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"chunklab {chunklab.__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version."
    ),
) -> None:
    """chunklab — find which chunking strategy retrieves your answers best."""


@app.command()
def run(
    docs: Path = typer.Option(..., "--docs", help="Document file or directory."),
    questions: Path = typer.Option(..., "--questions", help="questions.yaml path."),
    config: Path | None = typer.Option(None, "--config", help="config.yaml path (optional)."),
    out: Path | None = typer.Option(None, "--out", help="Output directory for reports."),
) -> None:
    """Run the chunking evaluation and write the reports."""
    from chunklab.config import load_config
    from chunklab.report.console import print_report
    from chunklab.report.html import write_html_report
    from chunklab.report.json_report import write_json_report
    from chunklab.runner import evaluate

    cfg = load_config(config)
    if out is not None:
        cfg.output.dir = str(out)

    with console.status("Running evaluation (first run downloads the embedding model)..."):
        report = evaluate(docs=docs, questions=questions, config=cfg)

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "console" in cfg.output.formats:
        print_report(report, console)
    if "html" in cfg.output.formats:
        html_path = write_html_report(report, out_dir / "report.html")
        console.print(f"\nHTML report: [bold]{html_path}[/bold]")
    if "json" in cfg.output.formats:
        json_path = write_json_report(report, out_dir / "report.json")
        console.print(f"JSON report: [bold]{json_path}[/bold]")


@app.command()
def validate(
    docs: Path = typer.Option(..., "--docs", help="Document file or directory."),
    questions: Path = typer.Option(..., "--questions", help="questions.yaml path."),
    config: Path | None = typer.Option(None, "--config", help="config.yaml path (optional)."),
) -> None:
    """Check a question set against the corpus, before spending a run on it.

    Exits non-zero when errors are found, so it can gate CI.
    """
    from chunklab.config import load_config, load_questions
    from chunklab.loaders.registry import load_documents
    from chunklab.validation import validate_questions

    cfg = load_config(config)
    documents = load_documents(docs)
    qs = load_questions(questions)
    report = validate_questions(qs, documents, fuzzy_threshold=cfg.eval.fuzzy_threshold)

    for issue in report.issues:
        tag = "[red]ERROR[/red]" if issue.severity == "error" else "[yellow]WARN [/yellow]"
        console.print(f"{tag} [bold]{issue.question_id}[/bold] ({issue.kind}): {issue.message}")
        if issue.suggestion:
            console.print(f"      found at [dim]{issue.location}[/dim], verbatim source text:")
            console.print(f"      [green]{issue.suggestion!r}[/green]")

    console.print(
        f"\n{report.num_questions} questions, {report.num_scored} scored, "
        f"{report.num_gold_snippets} gold snippets — "
        f"[red]{len(report.errors)} error(s)[/red], "
        f"[yellow]{len(report.warnings)} warning(s)[/yellow]"
    )
    if not report.ok:
        raise typer.Exit(1)
    console.print("[green]Question set is valid.[/green]")


@app.command()
def strategies() -> None:
    """List available chunking strategies and their default parameters."""
    from rich.table import Table

    from chunklab.chunkers.registry import available_strategies

    table = Table(title="Available chunking strategies")
    table.add_column("Name", style="bold")
    table.add_column("Description")
    for name, desc in available_strategies().items():
        table.add_row(name, desc)
    console.print(table)


@app.command()
def demo() -> None:
    """Launch the local Gradio demo app."""
    try:
        from app.app import build_app
    except ImportError:
        console.print("[red]The demo requires gradio:[/red] pip install 'chunklab[demo]'")
        raise typer.Exit(1) from None
    build_app().launch()


@app.command()
def bootstrap(
    docs: Path = typer.Option(..., "--docs", help="Document file or directory."),
    out: Path = typer.Option(
        Path("questions.draft.yaml"), "--out", help="Where to write the draft."
    ),
    n: int = typer.Option(20, "-n", help="How many questions to draft."),
    backend: str = typer.Option(
        "heuristic", "--backend", help="heuristic (default, fully local) | llm (not yet available)"
    ),
) -> None:
    """Draft a question set from your documents, to edit rather than start from scratch.

    Gold snippets are verbatim sentences; the queries are mechanical drafts and
    every question is marked `reviewed: false` until you have checked it.
    """
    from chunklab.eval.qa_gen import dump_questions_yaml, generate_questions
    from chunklab.loaders.registry import load_documents

    if backend != "heuristic":
        console.print(
            f"[red]Unknown backend '{backend}'.[/red] Only the local heuristic backend "
            "exists today; an optional LLM backend is planned and will stay off by default."
        )
        raise typer.Exit(2)

    documents = load_documents(docs)
    questions = generate_questions(documents, n=n)
    if not questions:
        console.print(
            "[red]No draftable sentences found.[/red] The heuristic looks for factual "
            "statements with a quantity, duration or amount; write questions by hand for "
            "this corpus (see docs/getting-started.md)."
        )
        raise typer.Exit(1)

    out.write_text(dump_questions_yaml(questions), encoding="utf-8")
    if len(questions) < n:
        console.print(
            f"[dim]Asked for {n}: the heuristic rejected the rest rather than emit "
            f"malformed questions. Add documents to draft more.[/dim]"
        )
    console.print(
        f"Wrote [bold]{len(questions)}[/bold] draft questions to [bold]{out}[/bold].\n"
        "[yellow]These are drafts.[/yellow] Rewrite each query in your users' words, "
        "delete the ones that are not worth asking, then set reviewed: true.\n"
        f"Check your edits with: chunklab validate --docs {docs} --questions {out}"
    )
