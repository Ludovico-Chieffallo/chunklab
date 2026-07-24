"""Guards the README against reintroducing unsourced quantitative claims.

Roadmap task 0.2 acceptance criterion: no number in the README without either
a linked source or a command in the repo that reproduces it. This test pins the
two specific unsourced stats that used to open the README so they cannot creep
back in unreferenced.
"""

import re
from pathlib import Path

README = Path(__file__).parent.parent / "README.md"


def _prose(text: str) -> str:
    """README minus fenced code blocks and HTML comments.

    Numbers inside example output blocks and questions.yaml snippets are
    reproducible/illustrative, not marketing claims, so they are out of scope.
    """
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    return text


def test_no_unsourced_retrieval_failure_stat():
    prose = _prose(README.read_text())
    # The old "failure is in retrieval roughly 70% of the time" claim.
    assert not re.search(r"70\s*%", prose), (
        "unsourced 70% retrieval-failure stat is back in the README prose"
    )


def test_no_unsourced_chunking_swing_stat():
    prose = _prose(README.read_text())
    # The old "~15-percentage-point swing" claim.
    assert not re.search(r"15[\s-]*percentage", prose, flags=re.IGNORECASE), (
        "unsourced 15-percentage-point swing stat is back in the README prose"
    )
