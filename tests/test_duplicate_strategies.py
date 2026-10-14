"""A strategy listed twice is refused, not silently merged.

The natural way to compare chunk sizes is to list `fixed` twice with two
`chunk_size` values. Results are keyed by strategy name, so the two entries
collided: on the bundled example the 64-token `fixed` was reported with the
512-token one's balanced score (0.42 against its own recall of 0.155), and the
verdict announced that `fixed` beat `fixed`. Until entries can carry a name of
their own, a repeated strategy is an error that says how to compare sizes.
"""

import pytest

from chunklab.config import Config
from chunklab.errors import InputError
from chunklab.models import Document, Question
from chunklab.runner import run_evaluation


def test_a_strategy_listed_twice_is_refused():
    config = Config.model_validate(
        {
            "embedding": {"backend": "fake"},
            "strategies": [
                {"name": "fixed", "params": {"chunk_size": 64, "overlap": 0}},
                {"name": "recursive"},
                {"name": "fixed", "params": {"chunk_size": 512, "overlap": 64}},
            ],
        }
    )
    document = Document(id="d", source_path="d.md", text="Invoices are payable within 45 days.")
    question = Question(id="q", query="?", gold_snippets=["Invoices are payable within 45 days."])

    with pytest.raises(InputError, match="'fixed' is listed 2 times") as caught:
        run_evaluation([document], [question], config)

    assert "separate runs" in str(caught.value)
