"""Refuse to publish when the pushed tag and the package version disagree.

release.yml publishes whatever a `v*` tag points at. A tag that does not match
`pyproject.toml` would publish under one number while every report.json the package
writes records another - on a number PyPI never lets you reuse.

Usage:
    python scripts/check_release_tag.py v0.4.1
"""

import sys
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: check_release_tag.py <tag>", file=sys.stderr)
        return 2
    tag = argv[1]
    version = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["version"]
    expected = f"v{version}"
    if tag != expected:
        print(
            f"tag {tag!r} does not match pyproject.toml version {version!r}; expected {expected!r}",
            file=sys.stderr,
        )
        return 1
    print(f"tag {tag} matches version {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
