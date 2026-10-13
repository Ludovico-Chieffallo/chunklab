"""What release.yml must check before anything reaches PyPI.

A version number on PyPI can never be reused, so these checks run before the upload,
on the artifact that would be uploaded - not on the checkout the test suite usually
sees. 0.4.0 passed its whole suite and still shipped a `chunklab demo` that could not
start, because nothing ever looked inside the wheel.
"""

import importlib
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent
SCRIPTS = ROOT / "scripts"
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


def _smoke():
    sys.path.insert(0, str(SCRIPTS))
    try:
        return importlib.import_module("smoke_test_wheel")
    finally:
        sys.path.remove(str(SCRIPTS))


def _check_tag(tag: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "check_release_tag.py"), tag],
        capture_output=True,
        text=True,
        check=False,
    )


def _wheel(tmp_path: Path, names: list[str]) -> Path:
    path = tmp_path / f"chunklab-{VERSION}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as wheel:
        for name in names:
            wheel.writestr(name, "")
    return path


# --- the tag ------------------------------------------------------------------------


def test_a_tag_matching_the_version_passes():
    result = _check_tag(f"v{VERSION}")

    assert result.returncode == 0, result.stderr


def test_a_tag_that_does_not_match_the_version_is_refused():
    """It would publish one number while the package reports another."""
    result = _check_tag("v9.9.9")

    assert result.returncode == 1
    assert "v9.9.9" in result.stderr and VERSION in result.stderr


def test_a_tag_without_the_v_prefix_is_refused():
    result = _check_tag(VERSION)

    assert result.returncode == 1


# --- the wheel's contents -----------------------------------------------------------------


def _complete() -> list[str]:
    return [*_smoke().REQUIRED, f"chunklab-{VERSION}.dist-info/METADATA"]


def test_a_complete_wheel_passes_the_contents_check(tmp_path):
    assert _smoke().wheel_problems(_wheel(tmp_path, _complete())) == []


def test_a_wheel_without_the_demo_is_refused(tmp_path):
    """The 0.4.0 wheel, exactly: no module for `chunklab demo` to import."""
    wheel = _wheel(tmp_path, [n for n in _complete() if n != "chunklab/demo.py"])

    assert any("chunklab/demo.py" in problem for problem in _smoke().wheel_problems(wheel))


def test_a_wheel_carrying_other_top_level_packages_is_refused(tmp_path):
    """A stray `tests/` or `app/` would install as a top-level module into every
    user's site-packages."""
    wheel = _wheel(tmp_path, [*_complete(), "tests/test_x.py", "app/app.py"])

    problems = " ".join(_smoke().wheel_problems(wheel))
    assert "tests" in problems and "app" in problems


@pytest.mark.slow
def test_the_wheel_built_from_this_checkout_passes_the_contents_check(tmp_path):
    """Builds for real (an isolated build fetches hatchling, hence slow)."""
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "-q", "-w", str(tmp_path), str(ROOT)],
        check=True,
    )
    (wheel,) = tmp_path.glob("chunklab-*.whl")

    assert _smoke().wheel_problems(wheel) == []


# --- the workflow --------------------------------------------------------------------


def _release_jobs() -> dict:
    return yaml.safe_load((ROOT / ".github" / "workflows" / "release.yml").read_text())["jobs"]


def _commands(job: dict) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


def test_the_release_verifies_the_tag_the_branch_and_the_suite_first():
    verify = _release_jobs()["verify"]

    commands = _commands(verify)
    assert "check_release_tag.py" in commands
    assert "merge-base --is-ancestor" in commands
    assert "pytest" in commands and "ruff check" in commands


def test_the_release_smoke_tests_the_built_wheel_before_publishing():
    jobs = _release_jobs()

    assert "verify" in _as_list(jobs["build"].get("needs"))
    assert "smoke_test_wheel.py" in _commands(jobs["build"])
    for name in ("publish-pypi", "publish-testpypi"):
        assert "build" in _as_list(jobs[name].get("needs")), name


def test_every_pull_request_smoke_tests_the_wheel_too():
    """So a packaging mistake is found on the PR that makes it, not at release."""
    ci = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())

    assert "smoke_test_wheel.py" in _commands(ci["jobs"]["package"])


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]
