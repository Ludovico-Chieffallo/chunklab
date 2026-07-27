"""Lexical tokenization for BM25.

Dense retrieval never sees words, so chunklab had no tokenizer until BM25 needed
one. The requirement is narrow: split text into comparable terms in every script
the loaders accept, without adding a dependency or a language setting.

Space-separated scripts get word tokens. Scripts written without spaces (Han,
Kana, Hangul, Thai) get **character bigrams** — the standard fallback for
CJK lexical search, and the reason a naive `\\w+` tokenizer is useless there: it
would emit one enormous token per sentence and BM25 would match nothing.
"""

import re

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

#: Ranges written without spaces between words.
_UNSEGMENTED = (
    (0x3040, 0x30FF),  # kana
    (0x3400, 0x4DBF),  # CJK extension A
    (0x4E00, 0x9FFF),  # CJK unified
    (0xAC00, 0xD7AF),  # hangul syllables
    (0xF900, 0xFAFF),  # CJK compatibility
    (0x0E00, 0x0E7F),  # thai
)


def _is_unsegmented(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in _UNSEGMENTED)


def tokenize(text: str) -> list[str]:
    """Lexical terms for BM25: lowercase words, plus bigrams for CJK runs."""
    terms: list[str] = []
    for word in _WORD_RE.findall(text.lower()):
        if any(_is_unsegmented(char) for char in word):
            # A run of unsegmented script: emit overlapping character bigrams,
            # falling back to the single character when the run is length 1.
            if len(word) == 1:
                terms.append(word)
            else:
                terms.extend(word[i : i + 2] for i in range(len(word) - 1))
        else:
            terms.append(word)
    return terms
