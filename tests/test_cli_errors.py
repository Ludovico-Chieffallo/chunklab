"""A mistake in what the user typed is reported in one line, not in a traceback.

Measured on 0.4.0 installed from PyPI: a mistyped path printed 38 lines of rich
traceback, a tab in questions.yaml 184, a misspelled strategy 41. The message that
said what to fix was the last line of each, under a stack trace through
chunklab's internals that no user needs.

The line is drawn at *input*: what the user wrote (paths, YAML, config values,
strategy parameters, a baseline file) fails as one line with exit code 2. A bug
in chunklab still raises with its traceback, because then the traceback is the
bug report - and `CHUNKLAB_DEBUG=1` brings it back for input errors too.
"""

import pytest

import chunklab.cli as cli_module
from chunklab.errors import InputError

DOC = (
    "# Billing\n\n"
    "Invoices are issued on the first business day of each month and are payable "
    "within 45 days of issue. Late payments accrue interest at 1.5% per month.\n"
)

QUESTIONS = (
    "questions:\n"
    "  - id: q1\n"
    '    query: "When are invoices payable?"\n'
    '    gold_snippets: ["payable within 45 days of issue"]\n'
)


@pytest.fixture()
def corpus(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "billing.md").write_text(DOC, encoding="utf-8")
    return docs


def _write(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _one_error_line(result) -> str:
    """The whole output, asserted to be a single 'Error:' line."""
    lines = [line for line in result.output.splitlines() if line.strip()]
    assert "Traceback" not in result.output, result.output
    assert len(lines) == 1, result.output
    assert lines[0].startswith("Error: "), result.output
    return lines[0]


# --- paths -------------------------------------------------------------------------


def test_a_missing_docs_path_is_one_line(tmp_path, cli):
    questions = _write(tmp_path, "q.yaml", QUESTIONS)

    result = cli("run", "--docs", tmp_path / "nope", "--questions", questions)

    assert result.exit_code == 2
    assert "docs path not found" in _one_error_line(result)


def test_a_missing_questions_file_is_one_line(corpus, tmp_path, cli):
    result = cli("run", "--docs", corpus, "--questions", tmp_path / "missing.yaml")

    assert result.exit_code == 2
    assert "questions file not found" in _one_error_line(result)


def test_a_missing_config_file_is_one_line(corpus, tmp_path, cli):
    questions = _write(tmp_path, "q.yaml", QUESTIONS)

    result = cli(
        "run", "--docs", corpus, "--questions", questions, "--config", tmp_path / "nope.yaml"
    )

    assert result.exit_code == 2
    assert "config file not found" in _one_error_line(result)


def test_validate_and_find_evidence_report_paths_the_same_way(tmp_path, cli):
    questions = _write(tmp_path, "q.yaml", QUESTIONS)

    for command in ("validate", "find-evidence"):
        result = cli(command, "--docs", tmp_path / "nope", "--questions", questions)

        assert result.exit_code == 2, command
        assert "docs path not found" in _one_error_line(result)


def test_bootstrap_reports_a_missing_docs_path(tmp_path, cli):
    result = cli("bootstrap", "--docs", tmp_path / "nope", "--out", tmp_path / "draft.yaml")

    assert result.exit_code == 2
    assert "docs path not found" in _one_error_line(result)


# --- what is inside the files -------------------------------------------------------


def test_invalid_yaml_names_the_line_and_column(corpus, tmp_path, cli):
    """A tab where YAML wants spaces: 184 lines of traceback in 0.4.0."""
    questions = _write(tmp_path, "q.yaml", "questions:\n\t- id: q1\n")

    result = cli("run", "--docs", corpus, "--questions", questions)

    assert result.exit_code == 2
    line = _one_error_line(result)
    assert "q.yaml" in line and "line 2" in line and "column 1" in line


def test_a_questions_file_without_a_questions_list_is_one_line(corpus, tmp_path, cli):
    questions = _write(tmp_path, "q.yaml", "- id: q1\n  query: what?\n")

    result = cli("run", "--docs", corpus, "--questions", questions)

    assert result.exit_code == 2
    assert "top-level 'questions:' list" in _one_error_line(result)


def test_a_malformed_question_is_named(corpus, tmp_path, cli):
    questions = _write(tmp_path, "q.yaml", "questions:\n  - id: q1\n    query: what?\n  - id: q2\n")

    result = cli("run", "--docs", corpus, "--questions", questions)

    assert result.exit_code == 2
    line = _one_error_line(result)
    assert "question 2" in line and "q2" in line and "query" in line


def test_questions_without_gold_snippets_are_one_line(corpus, tmp_path, cli, fake_config):
    questions = _write(tmp_path, "q.yaml", "questions:\n  - id: q1\n    query: what?\n")

    result = cli("run", "--docs", corpus, "--questions", questions, "--config", fake_config)

    assert result.exit_code == 2
    assert "no questions with gold snippets" in _one_error_line(result)


def test_an_unknown_strategy_is_one_line(corpus, tmp_path, cli):
    questions = _write(tmp_path, "q.yaml", QUESTIONS)
    config = _write(tmp_path, "c.yaml", "strategies:\n  - name: recursiv\n")

    result = cli("run", "--docs", corpus, "--questions", questions, "--config", config)

    assert result.exit_code == 2
    line = _one_error_line(result)
    assert "unknown strategy 'recursiv'" in line and "recursive" in line


def test_a_misspelled_strategy_parameter_is_one_line_and_lists_the_real_ones(corpus, tmp_path, cli):
    """`chunk_sise` raised a TypeError from inside the run, after every document
    had been loaded."""
    questions = _write(tmp_path, "q.yaml", QUESTIONS)
    config = _write(
        tmp_path,
        "c.yaml",
        "embedding:\n  backend: fake\n"
        "strategies:\n  - name: fixed\n    params: {chunk_sise: 256}\n",
    )

    result = cli("run", "--docs", corpus, "--questions", questions, "--config", config)

    assert result.exit_code == 2
    line = _one_error_line(result)
    assert "'fixed'" in line and "chunk_sise" in line
    assert "chunk_size" in line and "overlap" in line


def test_an_impossible_strategy_parameter_is_one_line(corpus, tmp_path, cli):
    questions = _write(tmp_path, "q.yaml", QUESTIONS)
    config = _write(
        tmp_path,
        "c.yaml",
        "embedding:\n  backend: fake\n"
        "strategies:\n  - name: fixed\n    params: {chunk_size: 64, overlap: 64}\n",
    )

    result = cli("run", "--docs", corpus, "--questions", questions, "--config", config)

    assert result.exit_code == 2
    assert "overlap must be smaller than chunk_size" in _one_error_line(result)


def test_the_unimplemented_openai_backend_is_refused_with_the_config(corpus, tmp_path, cli):
    """It used to fail with NotImplementedError only after every document had loaded."""
    questions = _write(tmp_path, "q.yaml", QUESTIONS)
    config = _write(tmp_path, "c.yaml", "embedding:\n  backend: openai\n")

    result = cli("run", "--docs", corpus, "--questions", questions, "--config", config)

    assert result.exit_code == 2
    line = _one_error_line(result)
    assert "openai" in line and "not available" in line


def test_an_unreadable_document_is_named(tmp_path, cli):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "broken.docx").write_bytes(b"not a zip archive")
    questions = _write(tmp_path, "q.yaml", QUESTIONS)

    result = cli("validate", "--docs", docs, "--questions", questions)

    assert result.exit_code == 2
    assert "broken.docx" in _one_error_line(result)


# --- check ---------------------------------------------------------------------------


def test_check_reads_the_baseline_before_spending_a_run(corpus, tmp_path, cli, monkeypatch):
    """A missing baseline used to be discovered after the whole evaluation."""
    import chunklab.runner

    def must_not_run(*args, **kwargs):
        raise AssertionError("evaluated before the baseline was read")

    monkeypatch.setattr(chunklab.runner, "evaluate", must_not_run)
    questions = _write(tmp_path, "q.yaml", QUESTIONS)

    result = cli(
        "check", "--docs", corpus, "--questions", questions, "--baseline", tmp_path / "b.json"
    )

    assert result.exit_code == 2
    line = _one_error_line(result)
    assert "baseline not found" in line and "--update-baseline" in line


def test_check_refuses_a_baseline_that_is_not_a_report(corpus, tmp_path, cli):
    questions = _write(tmp_path, "q.yaml", QUESTIONS)
    baseline = _write(tmp_path, "b.json", '{"hello": "world"')

    result = cli("check", "--docs", corpus, "--questions", questions, "--baseline", baseline)

    assert result.exit_code == 2
    assert "not a chunklab report" in _one_error_line(result)


# --- what must NOT be swallowed ----------------------------------------------------------


def test_a_bug_keeps_its_traceback(corpus, tmp_path, cli, monkeypatch):
    """Only input errors are shortened. Anything else is chunklab's fault, and
    the traceback is what makes it fixable."""
    import chunklab.runner

    def broken(*args, **kwargs):
        raise RuntimeError("an internal bug")

    monkeypatch.setattr(chunklab.runner, "evaluate", broken)
    questions = _write(tmp_path, "q.yaml", QUESTIONS)

    result = cli("run", "--docs", corpus, "--questions", questions)

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)


def test_chunklab_debug_restores_the_traceback_for_input_errors(tmp_path, cli, monkeypatch):
    monkeypatch.setenv("CHUNKLAB_DEBUG", "1")
    questions = _write(tmp_path, "q.yaml", QUESTIONS)

    result = cli("run", "--docs", tmp_path / "nope", "--questions", questions)

    assert isinstance(result.exception, InputError)


def test_tracebacks_never_print_local_variables():
    """Locals of a crashed run include document text, which then lands in CI logs."""
    assert cli_module.app.pretty_exceptions_show_locals is False


# --- the Python API keeps its exception types ---------------------------------------------


def test_input_errors_are_still_the_exceptions_api_callers_caught():
    from chunklab.config import load_questions
    from chunklab.loaders.registry import load_documents

    with pytest.raises(FileNotFoundError):
        load_documents("/definitely/not/here")
    with pytest.raises(ValueError):
        load_questions(__file__)  # Python source is not a questions file
