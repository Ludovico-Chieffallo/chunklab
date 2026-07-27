"""Shared helpers for the public-benchmark harness.

The datasets are *not* vendored into this repository: they are downloaded on
demand into a cache directory the user controls. Both are CC BY 4.0 and must be
attributed; see docs/benchmarks.md.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CACHE = Path.home() / ".cache" / "chunklab-benchmarks"


def download(url: str, destination: Path, expected_bytes: int | None = None) -> Path:
    """Fetch `url` to `destination` once; return the local path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    print(f"downloading {url}")
    tmp = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed, documented URLs
    if expected_bytes is not None and tmp.stat().st_size != expected_bytes:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{url}: unexpected size")
    tmp.rename(destination)
    return destination


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class ConversionStats:
    """What the conversion kept and what it threw away.

    Reported in full because the drop rate is the honest measure of how far the
    benchmark we run is from the benchmark as published.
    """

    documents: int = 0
    questions_seen: int = 0
    questions_kept: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    gold_snippets: int = 0

    def drop(self, reason: str, count: int = 1) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + count

    def render(self) -> str:
        lines = [
            f"documents:        {self.documents}",
            f"questions seen:   {self.questions_seen}",
            f"questions kept:   {self.questions_kept}"
            f" ({self.questions_kept / max(self.questions_seen, 1):.1%})",
            f"gold snippets:    {self.gold_snippets}",
        ]
        if self.dropped:
            lines.append("dropped:")
            for reason, count in sorted(self.dropped.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {count:>6}  {reason}")
        return "\n".join(lines)


def write_questions_yaml(path: Path, questions: list[dict], header: list[str]) -> None:
    """Write a questions.yaml with verbatim gold snippets."""
    lines = [*header, "questions:"]
    for question in questions:
        lines.append(f"  - id: {question['id']}")
        lines.append(f"    query: {_yaml_str(question['query'])}")
        lines.append("    gold_snippets:")
        for gold in question["gold_snippets"]:
            lines.append(f"      - {_yaml_str(gold)}")
        if question.get("tags"):
            lines.append(f"    tags: [{', '.join(question['tags'])}]")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _yaml_str(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'
