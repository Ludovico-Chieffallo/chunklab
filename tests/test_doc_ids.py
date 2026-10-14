"""Two files with the same name are two documents.

A document's id was its file name without the extension, so `2023/msa.md` and
`2024/msa.md` were both `msa`, and so were `report.pdf` and `report.docx`. Their
chunks then shared ids (`msa:fixed:0`): hybrid retrieval fuses rankings by chunk id
and returned one chunk where dense and bm25 each returned two, and the runner kept a
single document per id, so chunk health and the test for answers split across
adjacent chunks read the other file's text. Folders per year, a README.md in every
folder, the same report exported twice: real corpora are full of these.

Only colliding files change id, so every corpus without a collision keeps its ids and
its `corpus_sha256`.
"""

from pathlib import Path

import pytest

from chunklab.config import Config
from chunklab.errors import InputError
from chunklab.loaders.registry import load_documents
from chunklab.models import Document, Question
from chunklab.runner import evaluate, run_evaluation

EXAMPLES = Path(__file__).parent.parent / "examples"


def _write(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _ids(root: Path) -> list[str]:
    return [d.id for d in load_documents(root)]


def test_files_with_the_same_name_in_different_folders_get_distinct_ids(tmp_path):
    root = _write(tmp_path, {"2023/msa.md": "The 2023 terms.", "2024/msa.md": "The 2024 terms."})

    assert _ids(root) == ["2023/msa.md", "2024/msa.md"]


def test_the_same_name_with_different_extensions_gets_distinct_ids(tmp_path):
    root = _write(tmp_path, {"report.md": "Exported once.", "report.txt": "Exported twice."})

    assert _ids(root) == ["report.md", "report.txt"]


def test_only_the_colliding_files_change_id(tmp_path):
    root = _write(
        tmp_path,
        {"2023/msa.md": "a", "2024/msa.md": "b", "notes.md": "c", "2024/annex.txt": "d"},
    )

    assert sorted(_ids(root)) == ["2023/msa.md", "2024/msa.md", "annex", "notes"]


def test_a_stem_that_equals_a_disambiguated_id_is_disambiguated_too(tmp_path):
    """`x.md.txt` has the stem `x.md`, which is also the id `x.md` gets once it
    collides with `sub/x.md`."""
    root = _write(tmp_path, {"x.md": "a", "sub/x.md": "b", "x.md.txt": "c"})

    ids = _ids(root)
    assert len(set(ids)) == 3
    assert sorted(ids) == ["sub/x.md", "x.md", "x.md.txt"]


def test_the_bundled_example_keeps_its_ids():
    """No collision, so ids, chunk ids and corpus_sha256 of existing reports hold."""
    documents = load_documents(EXAMPLES / "corpus")

    assert [d.id for d in documents] == [Path(d.source_path).stem for d in documents]


def test_a_run_refuses_documents_that_share_an_id():
    """Documents built in code bypass the loader, so the runner checks too."""
    documents = [
        Document(id="msa", source_path="2023/msa.md", text="The 2023 terms run for a year."),
        Document(id="msa", source_path="2024/msa.md", text="The 2024 terms run for two years."),
    ]
    questions = [Question(id="q", query="?", gold_snippets=["The 2024 terms run for two years."])]
    config = Config.model_validate({"embedding": {"backend": "fake"}})

    with pytest.raises(InputError, match="'msa'"):
        run_evaluation(documents, questions, config)


def test_both_files_are_evaluated_as_separate_documents(tmp_path):
    body = "Section {n}. The {year} master services agreement renews every {term}.\n\n"
    files = {
        f"{year}/msa.md": "".join(body.format(n=n, year=year, term=term) for n in range(1, 6))
        for year, term in (("2023", "twelve months"), ("2024", "twenty-four months"))
    }
    root = _write(tmp_path / "docs", files)
    questions = [
        Question(
            id="q",
            query="How often does the 2024 agreement renew?",
            gold_snippets=["The 2024 master services agreement renews every twenty-four months"],
        )
    ]
    config = Config.model_validate({"embedding": {"backend": "fake"}})

    report = evaluate(root, questions, config)

    assert report.corpus_summary["documents"] == ["2023/msa.md", "2024/msa.md"]
    assert report.strategy_results
    for result in report.strategy_results:
        for question in result.per_question:
            chunk_ids = [rc.chunk.id for rc in question.retrieved]
            assert len(chunk_ids) == len(set(chunk_ids)), (result.strategy, result.retriever)
