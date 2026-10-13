"""The sdist must carry the project, and nothing that merely sits next to it.

hatchling's default sdist is "every file this repository's .gitignore does not
exclude". A file kept out of git by any other rule - a global excludes file, an
editor's or an assistant's local settings - therefore went into an sdist built from
a working tree: building 0.4.1 locally before tagging put
`.claude/settings.local.json` into the archive. The release workflow builds from a
clean checkout and was not affected, but one manual `python -m build` would have
published it, and PyPI does not let you take a file back.
"""

import shutil
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def test_the_sdist_is_built_from_an_explicit_list():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    sdist = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]

    included = sdist.get("only-include") or sdist.get("include")
    assert included, "the sdist must list what it ships"
    assert all(not entry.lstrip("/").startswith(".") for entry in included)


@pytest.mark.slow
def test_files_beside_the_project_stay_out_of_the_sdist(tmp_path):
    """Builds for real from a copy of the tracked files plus two planted ones."""
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    tree = tmp_path / "tree"
    for name in tracked:
        target = tree / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    (tree / ".claude").mkdir(exist_ok=True)
    (tree / ".claude" / "settings.local.json").write_text("{}", encoding="utf-8")
    (tree / "notes.private.md").write_text("not for publishing\n", encoding="utf-8")

    subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--outdir", str(tmp_path / "dist"), str(tree)],
        check=True,
        capture_output=True,
    )
    (archive,) = (tmp_path / "dist").glob("chunklab-*.tar.gz")
    with tarfile.open(archive) as sdist:
        names = {name.split("/", 1)[1] for name in sdist.getnames() if "/" in name}

    assert not any(name.startswith(".claude") for name in names)
    assert "notes.private.md" not in names
    for expected in ("pyproject.toml", "README.md", "LICENSE", "chunklab/cli.py", "tests"):
        assert any(n == expected or n.startswith(expected + "/") for n in names), expected
