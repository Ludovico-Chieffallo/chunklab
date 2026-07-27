"""Roadmap task 0.4: two runs on identical inputs produce identical reports."""

import json
import re
from pathlib import Path

import chunklab
from chunklab.config import default_config, load_questions
from chunklab.runner import run_evaluation

EXAMPLES = Path(__file__).parent.parent / "examples"
TEST_DATA = Path(__file__).parent / "data"


def _run(handbook):
    config = default_config()
    config.embedding.backend = "fake"
    questions = load_questions(TEST_DATA / "questions.example.yaml")
    return run_evaluation([handbook], questions, config)


def test_run_is_deterministic(handbook):
    d1 = json.loads(_run(handbook).model_dump_json())
    d2 = json.loads(_run(handbook).model_dump_json())
    assert d1.pop("generated_at") and d2.pop("generated_at")
    assert d1 == d2


def test_provenance_fields(handbook):
    cs = _run(handbook).corpus_summary
    assert cs["chunklab_version"] == chunklab.__version__
    assert re.fullmatch(r"[0-9a-f]{64}", cs["corpus_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", cs["questions_sha256"])
    assert cs["embedding_model_revision"] is None  # fake backend has no revision


def test_schema_version_present(handbook):
    report = _run(handbook)
    assert report.schema_version == "1.1"


def test_corpus_hash_is_doc_order_independent(handbook):
    from chunklab.runner import _corpus_sha256

    other = handbook.model_copy(update={"id": "zzz_other", "text": "Some other text."})
    assert _corpus_sha256([handbook, other]) == _corpus_sha256([other, handbook])
