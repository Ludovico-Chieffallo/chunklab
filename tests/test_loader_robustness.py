"""Loader failure modes that used to be silent or unattributable (roadmap 6.3)."""

import pytest

from chunklab.config import default_config
from chunklab.loaders.registry import DocumentLoadError, load_documents
from chunklab.loaders.text import TextLoader
from chunklab.models import Document, Question
from chunklab.runner import run_evaluation

BODY = "# Retention policy\n\nRecords are kept for thirty days.\n"


def test_byte_order_mark_does_not_erase_the_structure(tmp_path):
    """Regression: a BOM detached the first heading and dropped every element."""
    path = tmp_path / "bom.md"
    path.write_text(BODY, encoding="utf-8-sig")

    doc = TextLoader().load(path)

    assert not doc.text.startswith("﻿")
    assert [e.type for e in doc.elements] == ["heading"]
    assert doc.metadata["encoding"] == "utf-8-bom"


def test_plain_utf8_is_reported_as_utf8(tmp_path):
    path = tmp_path / "plain.md"
    path.write_text(BODY, encoding="utf-8")

    doc = TextLoader().load(path)

    assert doc.metadata["encoding"] == "utf-8"
    assert [e.type for e in doc.elements] == ["heading"]


def test_non_utf8_file_is_read_instead_of_crashing(tmp_path):
    """Regression: a cp1252 file raised UnicodeDecodeError and killed the run."""
    path = tmp_path / "latin.md"
    path.write_bytes("# Città\n\nIl canone è dovuto.\n".encode("cp1252"))

    doc = TextLoader().load(path)

    assert "Città" in doc.text
    assert "è dovuto" in doc.text
    assert doc.metadata["encoding"] == "cp1252"


def test_load_error_names_the_file(tmp_path):
    """Regression: a bad file raised a parser traceback with no path in it."""
    (tmp_path / "good.md").write_text(BODY, encoding="utf-8")
    (tmp_path / "broken.docx").write_bytes(b"not a docx at all")

    with pytest.raises(DocumentLoadError, match="broken.docx"):
        load_documents(tmp_path)


def _questions() -> list[Question]:
    return [
        Question(
            id="q1",
            query="How long are records kept?",
            gold_snippets=["Records are kept for thirty days"],
        )
    ]


def test_empty_document_is_reported_not_ignored():
    config = default_config()
    config.embedding.backend = "fake"
    documents = [
        Document(id="real", source_path="real.md", text=BODY, elements=[], metadata={}),
        Document(id="scanned", source_path="scan.pdf", text="  \n\n ", elements=[], metadata={}),
    ]

    report = run_evaluation(documents, _questions(), config)

    assert any("no extractable text" in w and "scanned" in w for w in report.warnings)


def test_a_corpus_with_no_text_at_all_is_an_error():
    config = default_config()
    config.embedding.backend = "fake"
    documents = [Document(id="scanned", source_path="s.pdf", text="", elements=[], metadata={})]

    with pytest.raises(ValueError, match="no document contains extractable text"):
        run_evaluation(documents, _questions(), config)


def test_fallback_encoding_is_surfaced_in_the_report():
    config = default_config()
    config.embedding.backend = "fake"
    documents = [
        Document(
            id="latin",
            source_path="latin.md",
            text=BODY,
            elements=[],
            metadata={"encoding": "cp1252"},
        )
    ]

    report = run_evaluation(documents, _questions(), config)

    assert any("not valid UTF-8" in w for w in report.warnings)
