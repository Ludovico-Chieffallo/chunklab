"""Install a built wheel into a fresh virtualenv and use it the way a user would.

The test suite runs from the checkout, which is exactly why it could not see that the
0.4.0 wheel had no module for `chunklab demo` to import: in the checkout, `app/` was
always there. This script trusts only what `pip install` actually put on disk, and
runs every command from a directory outside the repository, so nothing in the
checkout can stand in for what the wheel lacks.

It needs no embedding model: the evaluation it runs uses the offline `fake` backend.
Point pip at the CPU-only PyTorch index to keep the install small, e.g.
`PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu`.

Usage:
    python -m build --wheel
    python scripts/smoke_test_wheel.py dist/chunklab-*.whl --expect-version 0.4.1
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path

#: Modules a user reaches through a command. If one is missing from the wheel, that
#: command cannot work from a PyPI install whatever the tests say.
REQUIRED = (
    "chunklab/__init__.py",
    "chunklab/__main__.py",
    "chunklab/cli.py",
    "chunklab/demo.py",
    "chunklab/errors.py",
    "chunklab/runner.py",
)

DOC = (
    "# Retention\n\n"
    "Records are kept for thirty days unless a legal hold applies to them.\n\n"
    "# Billing\n\n"
    "Invoices are payable within forty-five days of the date they are issued.\n"
)
QUESTIONS = (
    "questions:\n"
    "  - id: q1\n"
    '    query: "How long are records kept?"\n'
    '    gold_snippets: ["Records are kept for thirty days unless a legal hold applies"]\n'
    "  - id: q2\n"
    '    query: "When are invoices payable?"\n'
    '    gold_snippets: ["payable within forty-five days of the date they are issued"]\n'
)


def wheel_problems(wheel: Path) -> list[str]:
    """What is wrong with the wheel's contents, before installing anything."""
    names = zipfile.ZipFile(wheel).namelist()
    problems = [f"missing {name}" for name in REQUIRED if name not in names]
    top_level = {name.split("/", 1)[0] for name in names}
    stray = sorted(
        entry
        for entry in top_level
        if entry != "chunklab" and not entry.endswith((".dist-info", ".data"))
    )
    problems += [f"unexpected top-level entry {entry}/" for entry in stray]
    return problems


def _bin(env: Path, name: str) -> Path:
    scripts = env / ("Scripts" if os.name == "nt" else "bin")
    return scripts / (f"{name}.exe" if os.name == "nt" else name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--expect-version", required=True)
    args = parser.parse_args()

    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"\n      {detail}" if not ok else ""))
        if not ok:
            failures.append(label)

    problems = wheel_problems(args.wheel)
    check("wheel contents", not problems, "; ".join(problems))

    with tempfile.TemporaryDirectory(prefix="chunklab-smoke-") as tmp:
        work = Path(tmp)
        env = work / "venv"
        venv.EnvBuilder(with_pip=True).create(env)
        python, chunklab = _bin(env, "python"), _bin(env, "chunklab")
        subprocess.run(
            [str(python), "-m", "pip", "install", "-q", str(args.wheel.resolve())], check=True
        )

        # Outside the checkout, and with nothing on PYTHONPATH that could point back
        # into it: only what the wheel installed may answer.
        clean_env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}
        }

        def run(*command: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                command, cwd=work, capture_output=True, text=True, check=False, env=clean_env
            )

        expected = f"chunklab {args.expect_version}"
        version = run(str(chunklab), "--version")
        check(
            "chunklab --version",
            version.stdout.strip() == expected,
            version.stdout + version.stderr,
        )
        module = run(str(python), "-m", "chunklab", "--version")
        check("python -m chunklab --version", module.stdout.strip() == expected, module.stderr)

        demo = run(str(python), "-c", "import chunklab.demo")
        check("the demo module imports", demo.returncode == 0, demo.stderr)

        (work / "docs").mkdir()
        (work / "docs" / "policy.md").write_text(DOC, encoding="utf-8")
        (work / "questions.yaml").write_text(QUESTIONS, encoding="utf-8")
        (work / "fake.yaml").write_text("embedding:\n  backend: fake\n", encoding="utf-8")

        validate = run(str(chunklab), "validate", "--docs", "docs", "--questions", "questions.yaml")
        check("chunklab validate", validate.returncode == 0, validate.stdout + validate.stderr)

        evaluation = run(
            str(chunklab),
            "run",
            "--docs",
            "docs",
            "--questions",
            "questions.yaml",
            "--config",
            "fake.yaml",
            "--out",
            "out",
        )
        report = work / "out" / "report.json"
        ok = (
            evaluation.returncode == 0
            and report.is_file()
            and (work / "out" / "report.html").is_file()
        )
        check("chunklab run writes both reports", ok, evaluation.stdout + evaluation.stderr)
        if report.is_file():
            data = json.loads(report.read_text(encoding="utf-8"))
            check(
                "report.json holds ranked results",
                bool(data.get("strategy_results")) and bool(data.get("schema_version")),
            )

        mistake = run(str(chunklab), "run", "--docs", "missing", "--questions", "questions.yaml")
        lines = [line for line in mistake.stdout.splitlines() if line.strip()]
        check(
            "an input mistake is one line with exit code 2",
            mistake.returncode == 2 and len(lines) == 1 and "Traceback" not in mistake.stdout,
            f"exit {mistake.returncode}: {mistake.stdout[:300]}",
        )

    print(f"\n{len(failures)} failure(s)" if failures else "\nthe installed wheel works")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
