"""Files nobody meant to evaluate must not stop the run.

Word writes `~$name.docx` next to every document that is open, and a run over a
folder containing one failed with 103 lines of PackageNotFoundError - for a file the
user cannot see in Finder and never chose. Hidden files and directories (`.git/`,
LibreOffice's `.~lock.name.docx#`, an editor's backup) are the same story.

They are skipped when a *directory* is scanned, and said so in one line. A file the
user names explicitly is always loaded: that was a choice.
"""

import shutil
from pathlib import Path

import pytest

from chunklab.loaders.registry import load_documents

EXAMPLES = Path(__file__).parent.parent / "examples" / "corpus"
BODY = "# Retention policy\n\nRecords are kept for thirty days unless a legal hold applies.\n"

QUESTIONS = (
    "questions:\n"
    "  - id: q1\n"
    '    query: "How long are records kept?"\n'
    '    gold_snippets: ["Records are kept for thirty days unless a legal hold applies"]\n'
)


@pytest.fixture()
def folder(tmp_path):
    """A real document, plus everything a working folder accumulates around it."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "policy.md").write_text(BODY, encoding="utf-8")
    shutil.copy(EXAMPLES / "policy_tables.docx", docs / "tables.docx")
    (docs / "~$tables.docx").write_bytes(b"\x0bLudovico" + b"\x00" * 150)  # Word owner file
    (docs / ".draft.md").write_text("# Scratch\n\nnot for evaluation\n", encoding="utf-8")
    (docs / ".git").mkdir()
    (docs / ".git" / "description.md").write_text("# git\n", encoding="utf-8")
    return docs


def test_a_word_lock_file_does_not_stop_the_load(folder):
    documents = load_documents(folder)

    assert sorted(d.id for d in documents) == ["policy", "tables"]


def test_skipped_files_are_reported_to_the_caller(folder):
    skipped: list[Path] = []

    load_documents(folder, skipped=skipped)

    assert sorted(p.name for p in skipped) == [".draft.md", "description.md", "~$tables.docx"]


def test_a_file_named_explicitly_is_always_loaded(folder):
    documents = load_documents(folder / ".draft.md")

    assert [d.id for d in documents] == [".draft"]


def test_run_says_which_files_it_skipped(folder, tmp_path, cli, fake_config):
    questions = tmp_path / "q.yaml"
    questions.write_text(QUESTIONS, encoding="utf-8")

    result = cli(
        "run",
        "--docs",
        folder,
        "--questions",
        questions,
        "--config",
        fake_config,
        "--out",
        tmp_path / "out",
    )

    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert "skipped 3 hidden or lock file(s)" in flat
    assert "~$tables.docx" in flat


def test_validate_says_which_files_it_skipped(folder, tmp_path, cli):
    questions = tmp_path / "q.yaml"
    questions.write_text(QUESTIONS, encoding="utf-8")

    result = cli("validate", "--docs", folder, "--questions", questions)

    assert result.exit_code == 0, result.output
    assert "skipped 3 hidden or lock file(s)" in " ".join(result.output.split())


def test_a_folder_holding_only_skipped_files_says_what_it_skipped(tmp_path):
    """'no supported documents' alone would be false: there were documents, all
    left out on purpose."""
    from chunklab.errors import InputError

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "~$contract.docx").write_bytes(b"\x0bLudovico" + b"\x00" * 150)

    with pytest.raises(InputError, match=r"skipped 1 hidden or lock file\(s\)"):
        load_documents(docs)
