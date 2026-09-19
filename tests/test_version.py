"""The version lives in two files and must not drift.

The previous test asserted a hard-coded literal, which verified nothing: it
duplicated the constant it was checking and forced a manual edit on every
release. What can actually go wrong is `pyproject.toml` and
`chunklab.__version__` disagreeing — the package would then publish under one
number while reporting another in every `report.json` it writes.
"""

import re
import tomllib
from pathlib import Path

import chunklab

PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"


def declared_version() -> str:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["version"]


def test_version_matches_pyproject():
    assert chunklab.__version__ == declared_version(), (
        "chunklab/__init__.py and pyproject.toml disagree about the version; "
        "reports would record a number the package was not published under"
    )


def test_version_is_a_release_number():
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:rc\d+)?", chunklab.__version__), (
        f"{chunklab.__version__!r} is not MAJOR.MINOR.PATCH — the release workflow "
        "routes tags containing 'rc' to TestPyPI and everything else to PyPI"
    )


def test_module_entry_point_runs():
    """`python -m chunklab` printed nothing and exited 0 before chunklab/__main__.py
    existed, which reads as a broken install. It is what people reach for when the
    console script is not on PATH — an unactivated virtualenv, or CI."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "chunklab", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "chunklab" in result.stdout


def test_find_evidence_is_listed_in_help():
    """A command nobody can discover is a command nobody uses."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "chunklab", "--help"],
        capture_output=True,
        text=True,
        check=False,
        env={"COLUMNS": "200", "PATH": "/usr/bin:/bin"},
    )

    assert result.returncode == 0, result.stderr
    assert "find-evidence" in result.stdout
