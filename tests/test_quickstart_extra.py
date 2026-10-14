"""Every command that runs the bundled example says to install what it needs.

0.5.0 moved PDF support into the `pdf` extra, and the bundled example corpus holds
a PDF. The README's 60-second quickstart still began from `pip install chunklab`, so
on PyPI the first command a new user ran stopped at "reading PDFs needs the optional
PDF extra". The suite never saw it: the development install has every extra.
"""

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
PAGES = ("README.md", "examples/README.md", "docs/metrics.md", "docs/getting-started.md")


def _blocks_running_the_example() -> list[tuple[str, str]]:
    found = []
    for page in PAGES:
        text = (ROOT / page).read_text(encoding="utf-8")
        for block in re.findall(r"```bash\n(.*?)```", text, re.S):
            if "--docs examples/corpus" in block:
                found.append((page, block))
    return found


def test_the_example_corpus_still_holds_a_pdf():
    """If this fails the extra may no longer be needed: update the pages, then this."""
    assert list((ROOT / "examples" / "corpus").glob("*.pdf"))


def test_every_command_running_the_example_installs_the_pdf_extra():
    blocks = _blocks_running_the_example()

    assert blocks, "no page runs the bundled example any more"
    for page, block in blocks:
        assert "chunklab[pdf]" in block, f"{page}: {block.strip()}"
