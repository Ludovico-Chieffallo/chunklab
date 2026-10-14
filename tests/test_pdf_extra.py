"""PDF support is an extra, because its library is AGPL.

chunklab is MIT, and `pip install chunklab` installed PyMuPDF with it: AGPL-3.0, or
a commercial licence from Artifex. An organisation that does not allow AGPL code
could only find that out from the metadata of what pip had already installed.
PDF support now lives in `chunklab[pdf]`, and a PDF met without it is one line that
says which extra to install and why it is separate.
"""

import tomllib
from pathlib import Path

import pytest

import chunklab.loaders.pdf as pdf_loader
from chunklab.errors import InputError
from chunklab.loaders.registry import load_documents

ROOT = Path(__file__).parent.parent


def _project() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_the_plain_install_does_not_pull_in_pymupdf():
    assert not [spec for spec in _project()["dependencies"] if "pymupdf" in spec.lower()]


def test_the_pdf_extra_does():
    assert any(
        spec.startswith("pymupdf4llm") for spec in _project()["optional-dependencies"]["pdf"]
    )


@pytest.fixture()
def without_pdf_support(monkeypatch):
    monkeypatch.setattr(pdf_loader, "_pdf_library_installed", lambda: False)


def test_a_pdf_without_the_extra_is_one_line_naming_it(tmp_path, without_pdf_support):
    (tmp_path / "notes.md").write_text("# Notes\n\nPlain text loads without any extra.\n")
    (tmp_path / "report.pdf").write_bytes(b"%PDF-1.4\n")

    with pytest.raises(InputError) as caught:
        load_documents(tmp_path)

    message = str(caught.value)
    assert "pip install 'chunklab[pdf]'" in message
    assert "AGPL" in message and "1 PDF" in message


def test_the_check_comes_before_any_file_is_read(tmp_path, without_pdf_support, monkeypatch):
    """A folder of PDFs should not be half-parsed before the error."""
    read = []
    monkeypatch.setattr(
        "chunklab.loaders.text.TextLoader.load", lambda self, path: read.append(path)
    )
    (tmp_path / "a.md").write_text("text")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4\n")

    with pytest.raises(InputError):
        load_documents(tmp_path)

    assert read == []


def test_the_loader_says_the_same_when_called_directly(tmp_path, without_pdf_support):
    """The demo builds its loader by extension and calls it directly."""
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF-1.4\n")

    with pytest.raises(InputError, match=r"chunklab\[pdf\]"):
        pdf_loader.PdfLoader().load(path)


def test_text_formats_never_need_the_extra(tmp_path, without_pdf_support):
    (tmp_path / "notes.md").write_text("# Notes\n\nPlain text loads without any extra.\n")

    assert [d.id for d in load_documents(tmp_path)] == ["notes"]


def test_with_the_extra_a_pdf_is_read(tmp_path):
    """The extra itself, end to end: PyMuPDF writes a PDF and the loader reads it."""
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Invoices are payable within 45 days of issue.")
    document.save(tmp_path / "billing.pdf")

    (loaded,) = load_documents(tmp_path)

    assert loaded.id == "billing"
    assert "payable within 45 days of issue" in loaded.text
