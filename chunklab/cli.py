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


@app.command("gen-questions")
def gen_questions() -> None:
    """Generate an eval set from your documents (not yet implemented)."""
    console.print(
        "[yellow]gen-questions is planned for v0.2.[/yellow] "
        "For now, write 10-20 questions with verbatim gold_snippets - see the README."
    )
    raise typer.Exit(1)
