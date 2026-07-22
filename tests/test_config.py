from pathlib import Path

import pytest

from chunklab.config import default_config, load_config, load_questions

EXAMPLES = Path(__file__).parent.parent / "examples"


def test_example_config_validates():
    config = load_config(EXAMPLES / "config.example.yaml")
    assert config.retrieval.top_k == 5
    assert [s.name for s in config.strategies] == [
        "fixed",
        "recursive",
        "semantic",
        "semantic_no_floor",
        "structure",
    ]


def test_default_config_no_file():
    config = default_config()
    assert config.embedding.backend == "local"
    assert len(config.strategies) == 5


def test_invalid_strategy_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("strategies:\n  - name: bogus\n")
    with pytest.raises(ValueError, match="bogus"):
        load_config(bad)


def test_load_questions():
    questions = load_questions(EXAMPLES / "questions.example.yaml")
    assert len(questions) == 19
    assert questions[0].gold_snippets
    # exactly one question (the dress-code one) has no gold snippets
    assert sum(1 for q in questions if not q.gold_snippets) == 1
