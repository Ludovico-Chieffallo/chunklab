"""Tokenization and sentence-splitting invariants (roadmap 6.1).

`token_spans` maps byte-level tiktoken output back onto character offsets, which
is where non-ASCII text breaks: a token boundary can fall inside a multi-byte
character. These tests pin both the mapping and its cost.
"""

import time

import pytest

from chunklab.text_utils import count_tokens, sentence_spans, token_spans

TRICKY = [
    "The quick brown fox. Payment is due in 30 days!",
    "Il canone è dovuto entro trenta giorni — così è più però città sarà.",
    "契約は三十日以内に支払われるものとし、遅延利息が発生します。",
    "Ship it 🚀 now — 100% done ✅ già fatto",
    "Hello 世界 café 🚀 naïve résumé\n\nSecond paragraph, 42 GB.",
    "Данные хранятся тридцать дней.",
    "\n\n   \n",
]


@pytest.mark.parametrize("text", TRICKY)
def test_token_spans_tile_the_whole_text(text):
    """Spans start at 0, end at len(text), and never leave a gap between them."""
    spans = token_spans(text)
    assert len(spans) == count_tokens(text)
    if not spans:
        return
    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)
    for (start, end), (next_start, _) in zip(spans, spans[1:], strict=False):
        assert start <= end, "span is inverted"
        assert next_start >= start, "spans went backwards"
        assert next_start <= end, "gap between consecutive tokens"


@pytest.mark.parametrize("text", TRICKY)
def test_token_spans_are_valid_char_boundaries(text):
    """Every offset must slice cleanly - never inside a character."""
    spans = token_spans(text)
    for start, end in spans:
        assert 0 <= start <= end <= len(text)
        text[start:end]  # would raise only on an invalid index, but pins intent


def test_token_spans_stays_linear_on_multibyte_text():
    """Regression: scanning the byte->char map made this quadratic.

    48k chars of Japanese took ~19 s; nearly every token boundary splits a
    character there, so the fallback scan ran on almost every token.
    """
    text = "契約は三十日以内に支払われるものとし、遅延利息が発生します。" * 800
    token_spans("warm up the encoder")

    started = time.perf_counter()
    spans = token_spans(text)
    elapsed = time.perf_counter() - started

    assert len(spans) == count_tokens(text)
    # ~11 ms when linear, ~4.6 s when quadratic: any bound in between catches it.
    assert elapsed < 2.0, f"token_spans took {elapsed:.2f}s on {len(text)} chars"


def test_sentence_spans_end_at_markdown_emphasis():
    """Regression: '**Question?** Answer' was one sentence, gluing FAQ pairs."""
    text = "**How long are logs kept?** Logs are kept for thirty days."
    spans = sentence_spans(text)
    assert len(spans) == 2
    assert text[spans[0][0] : spans[0][1]] == "**How long are logs kept?**"


def test_sentence_spans_do_not_split_on_abbreviations():
    text = "Contact Dr. Rossi before the renewal. He signs the addendum."
    spans = sentence_spans(text)
    assert len(spans) == 2
