"""Regenerate the README quickstart example from a real run (roadmap task 1.4).

The block between the BEGIN/END markers in README.md is produced by this
script and must never be edited by hand. `--check` re-runs the evaluation and
exits non-zero if the README block no longer matches the current output.

Usage:
    python scripts/regen_readme_example.py          # rewrite the block
    python scripts/regen_readme_example.py --check  # verify, for CI
"""

import argparse
import re
import sys
from pathlib import Path

from rich.console import Console

REPO = Path(__file__).parent.parent
README = REPO / "README.md"
BEGIN = "<!-- BEGIN GENERATED EXAMPLE (scripts/regen_readme_example.py) -->"
END = "<!-- END GENERATED EXAMPLE -->"


def generated_block() -> str:
    from chunklab.report.console import print_report
    from chunklab.runner import evaluate

    report = evaluate(
        docs=REPO / "examples" / "corpus",
        questions=REPO / "examples" / "questions.yaml",
    )
    console = Console(record=True, width=100, force_terminal=False, no_color=True)
    print_report(report, console)
    text = console.export_text().strip("\n")
    return f"{BEGIN}\n```\n{text}\n```\n{END}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify instead of rewriting")
    args = parser.parse_args()

    readme = README.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(readme):
        sys.exit(f"markers not found in {README}")

    block = generated_block()
    updated = pattern.sub(lambda _: block, readme)

    if args.check:
        if updated != readme:
            sys.exit("README example block is stale: run scripts/regen_readme_example.py")
        print("README example block is up to date.")
        return

    README.write_text(updated, encoding="utf-8")
    print("README example block regenerated.")


if __name__ == "__main__":
    main()
