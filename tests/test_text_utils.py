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


def _full_prefix_sentence_spans(text: str) -> list[tuple[int, int]]:
    """The original splitter, re-reading the whole prefix at every boundary.

    Kept as the oracle for the windowed version: the optimisation is only
    allowed if it never changes a single span.
    """
    import re

    from chunklab.text_utils import _ABBREVIATIONS, _SENTENCE_END_RE

    boundaries = [0]
    for m in _SENTENCE_END_RE.finditer(text):
        before = text[: m.start() + 1]
        if re.split(r"[\s(]", before.rstrip().rstrip(".!?"))[-1].lower() in _ABBREVIATIONS:
            continue
        boundaries.append(m.end())
    boundaries.append(len(text))

    spans = []
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        segment = text[start:end]
        stripped = segment.strip()
        if not stripped:
            continue
        begin = start + len(segment) - len(segment.lstrip())
        spans.append((begin, begin + len(stripped)))
    return spans


ABBREVIATION_TRAPS = [
    # Every abbreviation, far enough into the text that a windowed lookback is
    # reading a slice rather than the whole string.
    "Filler sentence to push the offset along. " * 30 + "Contact Dr. Rossi today. Done.",
    "Padding here. " * 60 + "See Fig. 4 for the curve. It rises.",
    "Lead in. " * 50 + "Use e.g. the default value. Then stop.",
    "Words words. " * 40 + "Ship to St. Louis by Friday. Confirmed.",
    "Intro. " * 70 + "Volume no. 12 is missing. Reorder it.",
    # A token longer than the lookback window, so the slice cuts mid-word.
    "Prelude text. " * 20 + "See " + "x" * 200 + ". Next sentence here.",
    # A boundary inside the first window-length of the document.
    "Dr. Rossi signed. Then he left.",
    # Abbreviation immediately after an opening bracket, which the splitter
    # treats as a word separator.
    "Some lead in text here. (e.g. this parenthetical). And on.",
]


@pytest.mark.parametrize("text", ABBREVIATION_TRAPS + TRICKY)
def test_windowed_lookback_matches_reading_the_whole_prefix(text):
    """Regression: the lookback was a slice from offset 0, making the splitter
    quadratic — 2.9 s on one real 10-K, 154 s on a 12,000-sentence document.

    Windowing it is only safe if the decision never changes. Verified on 33 real
    documents (26 CRISPR papers, the example corpus, an Apple 10-K and a
    Microsoft annual report) with zero differing spans; these cases pin the
    reasoning that made it safe.
    """
    assert sentence_spans(text) == _full_prefix_sentence_spans(text)


def test_sentence_spans_stays_linear_on_a_long_document():
    """~0.06 s windowed against ~154 s quadratic: any bound in between catches
    a regression, and this one leaves two orders of magnitude of headroom."""
    text = " ".join(
        f"This is sentence number {i} about topic {i % 7}, e.g. the Dr. Smith case."
        for i in range(12_000)
    )
    sentence_spans("warm up")

    started = time.perf_counter()
    spans = sentence_spans(text)
    elapsed = time.perf_counter() - started

    assert len(spans) == 12_000
    assert elapsed < 5.0, f"sentence_spans took {elapsed:.2f}s on {len(text)} chars"
