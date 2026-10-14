"""An evaluation with no strategy in it is a mistake, not an empty report.

A config file without `strategies:` gets the default five, but that happened only in
`load_config`: a `Config` built in code started with none, and `run_evaluation`
returned a report with no results and no complaint. A test written against the
library fell into it - it checked every result of an empty list, and so checked
nothing.
"""

import pytest

from chunklab.config import Config, default_strategies
from chunklab.errors import InputError
from chunklab.models import Document, Question
from chunklab.runner import run_evaluation


def test_a_config_built_in_code_has_the_default_strategies():
    assert Config().strategies == default_strategies()
    assert Config.model_validate({"embedding": {"backend": "fake"}}).strategies == (
        default_strategies()
    )


def test_a_run_with_no_strategies_is_refused():
    config = Config.model_validate({"embedding": {"backend": "fake"}, "strategies": []})
    document = Document(id="d", source_path="d.md", text="Invoices are payable within 45 days.")
    question = Question(id="q", query="?", gold_snippets=["Invoices are payable within 45 days."])

    with pytest.raises(InputError, match="no strategies"):
        run_evaluation([document], [question], config)
