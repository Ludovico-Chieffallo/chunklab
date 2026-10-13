"""Text from the user's documents must reach the terminal as text, never as markup.

rich reads `[word]` as a style tag and `[/word]` as the end of one. Every message
chunklab printed was an f-string with document text, gold snippets, file paths or
strategy names pasted straight in, so whatever the corpus contained was parsed as
markup. The Llama 2 prompt format, `[INST] ... [/INST]`, is ordinary content in a
RAG corpus, and it was enough to crash `find-evidence`, and to crash `run` after the
whole evaluation had finished, before a single report had been written.
"""

import sys

from rich.console import Console

from chunklab.models import ChunkHealth, EvalReport, StrategyResult
from chunklab.report.console import print_report

PROMPT_DOC = (
    "# Prompt templates\n\n"
    "The Llama 2 chat format wraps every user turn as [INST] your question here [/INST] "
    "and the system prompt sits inside the first instruction block. Retention of chat "
    "transcripts is limited to 30 days for every workspace.\n\n"
    "## Billing\n\n"
    "Invoices are issued on the first business day of each month and are payable "
    "within 45 days of issue.\n"
)


def _flat(text: str) -> str:
    """Whitespace-normalised output, so a wrapped line still matches."""
    return " ".join(text.split())


def _corpus(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "prompts.md").write_text(PROMPT_DOC, encoding="utf-8")
    return docs


def _questions(tmp_path, body: str):
    path = tmp_path / "questions.yaml"
    path.write_text(body, encoding="utf-8")
    return path


# --- commands that print document text --------------------------------------------


def test_find_evidence_prints_a_passage_holding_a_closing_tag(tmp_path, cli):
    """Crashed with MarkupError: closing tag '[/INST]' doesn't match any open tag."""
    questions = _questions(
        tmp_path,
        'questions:\n  - id: q1\n    query: "How does the Llama 2 chat format wrap a user turn?"\n',
    )

    result = cli("find-evidence", "--docs", _corpus(tmp_path), "--questions", questions)

    assert result.exit_code == 0, result.output
    assert "[INST] your question here [/INST]" in _flat(result.output)


def test_validate_prints_a_suggestion_holding_a_closing_tag(tmp_path, cli):
    """A drifted snippet is answered with the verbatim source text, ready to paste.
    When that text held `[/INST]` the suggestion itself crashed the command."""
    questions = _questions(
        tmp_path,
        "questions:\n"
        "  - id: q1\n"
        '    query: "How does the Llama 2 chat format wrap a user turn?"\n'
        "    gold_snippets:\n"
        '      - "wraps every user turn as [INST] your questions here [/INST] and the system"\n',
    )

    result = cli("validate", "--docs", _corpus(tmp_path), "--questions", questions)

    assert result.exit_code == 0, result.output
    assert "fuzzy_only" in result.output
    assert "[INST] your question here [/INST]" in _flat(result.output)


# --- `run`: the evaluation is the expensive part, so it must never be lost -----------


def test_run_survives_a_warning_that_quotes_a_closing_tag(tmp_path, cli, fake_config):
    """A gold snippet under five tokens is quoted back in a warning. This one
    crashed `run` after the evaluation, and the output directory stayed empty."""
    questions = _questions(
        tmp_path,
        "questions:\n"
        "  - id: q1\n"
        '    query: "What closes an instruction block?"\n'
        '    gold_snippets: ["[/INST] and"]\n'
        "  - id: q2\n"
        '    query: "When are invoices payable?"\n'
        '    gold_snippets: ["payable within 45 days of issue"]\n',
    )
    out = tmp_path / "out"

    result = cli(
        "run",
        "--docs",
        _corpus(tmp_path),
        "--questions",
        questions,
        "--config",
        fake_config,
        "--out",
        out,
    )

    assert result.exit_code == 0, result.output
    assert "'[/INST] and'" in _flat(result.output)
    assert (out / "report.json").is_file()
    assert (out / "report.html").is_file()


def test_run_writes_the_reports_before_printing_to_the_console(
    tmp_path, cli, fake_config, monkeypatch
):
    """Whatever goes wrong while printing, the files must already be on disk: they
    are the result of the run, and the console is only a view of them."""
    import chunklab.report.console as console_report

    def broken_print_report(report, console=None):
        raise RuntimeError("terminal went away")

    monkeypatch.setattr(console_report, "print_report", broken_print_report)
    questions = _questions(
        tmp_path,
        'questions:\n  - id: q1\n    query: "When are invoices payable?"\n'
        '    gold_snippets: ["payable within 45 days of issue"]\n',
    )
    out = tmp_path / "out"

    result = cli(
        "run",
        "--docs",
        _corpus(tmp_path),
        "--questions",
        questions,
        "--config",
        fake_config,
        "--out",
        out,
    )

    assert result.exit_code != 0
    assert (out / "report.json").is_file()
    assert (out / "report.html").is_file()


# --- everything else that interpolates user-controlled text -------------------------


def test_cache_prints_a_path_holding_brackets(tmp_path, cli, monkeypatch):
    """`[draft]` in a directory name was swallowed as a style tag, printing a
    path that does not exist."""
    cache_dir = tmp_path / "[draft]"
    monkeypatch.setenv("CHUNKLAB_CACHE_DIR", str(cache_dir))

    result = cli("cache")

    assert result.exit_code == 0, result.output
    assert "[draft]" in result.output


def test_demo_install_hint_names_the_extra(cli, monkeypatch):
    """rich ate `[demo]` and told people to `pip install 'chunklab'`, which they
    already had — a loop with no way out."""
    monkeypatch.setitem(sys.modules, "gradio", None)
    monkeypatch.setitem(sys.modules, "app.app", None)

    result = cli("demo")

    assert result.exit_code != 0
    assert "chunklab[demo]" in result.output


def test_console_report_prints_names_and_warnings_literally():
    """Strategy names come from plugins, warnings quote gold snippets: both are
    data, and must render exactly as written."""
    health = ChunkHealth(
        num_chunks=3,
        tokens_min=10,
        tokens_median=20.0,
        tokens_mean=20.0,
        tokens_max=30,
        pct_tiny=0.0,
        pct_oversized=0.0,
        boundary_health=1.0,
    )
    result = StrategyResult(
        strategy="clauses[/v2]",
        recall_at_k=0.5,
        hit_rate_at_k=0.5,
        mrr=0.5,
        precision_at_k=0.1,
        chunk_health=health,
    )
    report = EvalReport(
        corpus_summary={"num_documents": 1, "num_scored_questions": 2, "top_k": 5},
        strategy_results=[result],
        recommendation="Use CLAUSES[/V2] chunking.",
        warnings=["1 gold snippet(s) are short (q1 ('[/INST] and'))"],
    )
    console = Console(record=True, width=200, force_terminal=False)

    print_report(report, console)

    text = console.export_text()
    assert "clauses[/v2]" in text
    assert "CLAUSES[/V2]" in text
    assert "'[/INST] and'" in text
