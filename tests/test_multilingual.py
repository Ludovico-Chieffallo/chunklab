"""Non-English corpora: segmentation, model prefixes, and mismatch warnings (roadmap 4).

The failure this phase targets is silent: an English-only model over an Italian
corpus, or a Japanese document that segments into a single sentence, both produce
a confident-looking report full of meaningless numbers.
"""

import pytest

from chunklab.config import default_config
from chunklab.embeddings.prefixes import scheme_for
from chunklab.language import (
    detect_language,
    dominant_script,
    model_language_scope,
)
from chunklab.models import Document, Question
from chunklab.runner import run_evaluation
from chunklab.text_utils import sentence_spans

ITALIAN = (
    "Il presente contratto disciplina la fornitura dei servizi. Le fatture sono pagate "
    "entro trenta giorni dalla data di emissione. Il fornitore garantisce che i dati "
    "sono conservati per un periodo non superiore a dodici mesi, salvo diverso accordo "
    "tra le parti. Nel caso in cui il cliente non provveda al pagamento, si applicano "
    "gli interessi di mora previsti dalla legge vigente."
)
JAPANESE = (
    "データは三十日間保管されます。請求書は三十日以内に支払われます。"
    "サポートは一時間以内に応答します。"
)


# --- sentence segmentation ----------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Records are kept for thirty days. Invoices are due in 30 days. Support replies.", 3),
        (JAPANESE, 3),
        ("数据保存三十天。发票须在三十天内支付。支持在一小时内响应。", 3),
        ("डेटा तीस दिनों तक रखा जाता है। चालान तीस दिनों में देय है। सहायता उत्तर देती है।", 3),
        ("هل تحفظ البيانات؟ تدفع الفواتير خلال ثلاثين يوما.", 2),
    ],
)
def test_sentence_terminators_of_other_scripts(text, expected):
    """Regression: a whole CJK or Devanagari document was one 'sentence'."""
    assert len(sentence_spans(text)) == expected


def test_ascii_terminators_still_need_whitespace():
    """The non-ASCII rule must not make '3.14' or 'e.g.' split."""
    assert len(sentence_spans("The rate is 3.14 percent per e.g. annum.")) == 1


def test_quoted_cjk_sentence_is_not_split_at_the_quote():
    assert len(sentence_spans("彼は「三十日です。」と述べた。次に請求書を送った。")) == 2


# --- model prefixes -----------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "query", "passage"),
    [
        ("intfloat/multilingual-e5-small", "query: ", "passage: "),
        ("intfloat/e5-base-v2", "query: ", "passage: "),
        ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
        ("BAAI/bge-m3", "", ""),
        ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
    ],
)
def test_prefix_scheme_per_model_family(model, query, passage):
    scheme = scheme_for(model)
    assert scheme.query == query
    assert scheme.passage == passage


def test_queries_and_passages_are_embedded_differently(monkeypatch):
    """E5 without prefixes silently loses quality; the embedder must add them."""
    from chunklab.embeddings.local import LocalEmbedder

    seen: list[list[str]] = []

    class FakeST:
        def __init__(self, *_args, **_kwargs):
            self.max_seq_length = 512

        def encode(self, texts, **_kwargs):
            import numpy as np

            seen.append(list(texts))
            return np.ones((len(texts), 4), dtype="float32")

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", FakeST)
    embedder = LocalEmbedder("intfloat/multilingual-e5-small")

    embedder.embed(["a chunk"])
    embedder.embed_queries(["a question"])

    assert seen == [["passage: a chunk"], ["query: a question"]]


def test_cache_does_not_mix_queries_with_passages(tmp_path):
    """The same text has two different vectors on the two sides of an asymmetric model."""
    from chunklab.embeddings.cache import CachedEmbedder, EmbeddingCache
    from tests.conftest import CountingEmbedder

    store = EmbeddingCache(tmp_path / "v.sqlite3")
    inner = CountingEmbedder()
    embedder = CachedEmbedder(inner, store)

    embedder.embed(["same text"])
    embedder.embed_queries(["same text"])

    assert len(inner.calls) == 2, "a query was served a passage vector"
    store.close()


def test_cache_separates_prefixed_from_unprefixed(tmp_path):
    """Turning prefixes off must not reuse vectors computed with them."""
    from chunklab.embeddings.cache import CachedEmbedder, EmbeddingCache
    from tests.conftest import CountingEmbedder

    store = EmbeddingCache(tmp_path / "v.sqlite3")

    with_prefix = CountingEmbedder(cache_signature="q=query: |p=passage: ")
    CachedEmbedder(with_prefix, store).embed(["same text"])

    without = CountingEmbedder(cache_signature="q=|p=")
    CachedEmbedder(without, store).embed(["same text"])

    assert without.calls == [["same text"]]
    store.close()


# --- language detection and the mismatch warning ------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [(ITALIAN, "it"), ("Records are kept for thirty days. " * 8, "en")],
)
def test_detects_latin_script_languages(text, expected):
    assert detect_language(text) == expected


def test_refuses_to_guess_from_too_little_text():
    assert detect_language("Records are kept for thirty days.") is None


@pytest.mark.parametrize(
    ("text", "script"),
    [
        (JAPANESE, "kana"),
        (ITALIAN, "latin"),
        ("Данные хранятся тридцать дней и потом удаляются.", "cyrillic"),
    ],
)
def test_dominant_script(text, script):
    assert dominant_script(text) == script


@pytest.mark.parametrize(
    ("model", "scope"),
    [
        ("BAAI/bge-small-en-v1.5", "english"),
        ("sentence-transformers/all-MiniLM-L6-v2", "english"),
        ("intfloat/multilingual-e5-small", "multilingual"),
        ("BAAI/bge-m3", "multilingual"),
        ("sentence-transformers/LaBSE", "multilingual"),
        ("some/unfamiliar-model", "unknown"),
    ],
)
def test_model_language_scope(model, scope):
    assert model_language_scope(model) == scope


def _run(model: str, text: str):
    config = default_config()
    config.embedding.backend = "fake"
    config.embedding.model = model
    documents = [Document(id="doc", source_path="doc.md", text=text, elements=[], metadata={})]
    questions = [
        Question(id="q1", query="quanto durano i dati?", gold_snippets=[text.split(".")[0]])
    ]
    return run_evaluation(documents, questions, config)


def test_english_model_on_italian_corpus_warns():
    report = _run("BAAI/bge-small-en-v1.5", ITALIAN)
    assert any("English-only" in w and "it" in w for w in report.warnings)


def test_multilingual_model_on_italian_corpus_is_silent():
    report = _run("intfloat/multilingual-e5-small", ITALIAN)
    assert not any("English-only" in w for w in report.warnings)


def test_english_model_on_english_corpus_is_silent():
    report = _run("BAAI/bge-small-en-v1.5", "Records are kept for thirty days. " * 8)
    assert not any("English-only" in w for w in report.warnings)


def test_unknown_model_makes_no_claim():
    """A wrong warning is worse than no warning."""
    report = _run("some/unfamiliar-model", ITALIAN)
    assert not any("English-only" in w for w in report.warnings)


def test_detected_languages_are_recorded_in_the_report():
    report = _run("intfloat/multilingual-e5-small", ITALIAN)
    assert report.corpus_summary["detected_languages"] == {"doc": "it"}
