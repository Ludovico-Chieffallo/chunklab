"""Script and language detection, to catch a corpus/model mismatch.

Running an English-only embedding model over an Italian or Japanese corpus does
not fail — it just retrieves badly, and every number in the report is quietly
worthless. That is the failure chunklab exists to surface, so it has to surface
it about itself.

Detection is deliberately conservative and dependency-free: a Unicode-range
count for the script, and stop-word frequency for Latin-script languages. When
the evidence is weak the answer is `None` and nothing is claimed. It only ever
drives a warning, never a decision.
"""

import re
from typing import Literal

_LETTER_RANGES: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    ("han", ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF))),
    ("kana", ((0x3040, 0x309F), (0x30A0, 0x30FF))),
    ("hangul", ((0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F))),
    ("cyrillic", ((0x0400, 0x052F),)),
    ("arabic", ((0x0600, 0x06FF), (0x0750, 0x077F))),
    ("devanagari", ((0x0900, 0x097F),)),
    ("hebrew", ((0x0590, 0x05FF),)),
    ("greek", ((0x0370, 0x03FF), (0x1F00, 0x1FFF))),
    ("thai", ((0x0E00, 0x0E7F),)),
    ("latin", ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F))),
)

LATIN_SCRIPT = "latin"


def dominant_script(text: str) -> str | None:
    """The script most letters belong to, or None when there are too few."""
    counts: dict[str, int] = {}
    for char in text:
        code = ord(char)
        for name, ranges in _LETTER_RANGES:
            if any(low <= code <= high for low, high in ranges):
                counts[name] = counts.get(name, 0) + 1
                break
    if not counts or sum(counts.values()) < 20:
        return None
    return max(counts, key=lambda k: counts[k])


#: Short, high-frequency function words. Enough to separate these languages on a
#: paragraph; not enough for a single sentence, which is why `min_words` exists.
_STOPWORDS: dict[str, set[str]] = {
    "en": {
        "the",
        "of",
        "and",
        "to",
        "in",
        "is",
        "that",
        "for",
        "it",
        "with",
        "as",
        "are",
        "be",
        "this",
        "on",
        "by",
        "from",
        "or",
        "an",
        "at",
        "not",
        "which",
        "have",
        "has",
        "was",
        "were",
        "will",
        "any",
        "all",
        "each",
    },
    "it": {
        "il",
        "lo",
        "la",
        "gli",
        "le",
        "di",
        "che",
        "per",
        "un",
        "una",
        "non",
        "con",
        "del",
        "della",
        "sono",
        "alla",
        "nel",
        "come",
        "più",
        "anche",
        "quando",
        "dal",
        "se",
        "ma",
        "si",
        "delle",
        "dei",
        "essere",
        "viene",
    },
    "es": {
        "el",
        "la",
        "los",
        "las",
        "de",
        "que",
        "en",
        "un",
        "una",
        "por",
        "con",
        "para",
        "del",
        "se",
        "no",
        "es",
        "son",
        "como",
        "más",
        "pero",
        "su",
        "al",
        "lo",
        "sus",
        "este",
        "esta",
    },
    "fr": {
        "le",
        "la",
        "les",
        "de",
        "des",
        "et",
        "un",
        "une",
        "dans",
        "pour",
        "que",
        "est",
        "en",
        "du",
        "au",
        "aux",
        "sur",
        "par",
        "ne",
        "pas",
        "plus",
        "ce",
        "cette",
        "sont",
        "qui",
        "ou",
    },
    "de": {
        "der",
        "die",
        "das",
        "und",
        "den",
        "von",
        "zu",
        "mit",
        "ist",
        "im",
        "für",
        "ein",
        "eine",
        "nicht",
        "auch",
        "auf",
        "dem",
        "des",
        "sich",
        "werden",
        "sind",
        "als",
        "es",
        "bei",
        "oder",
    },
    "pt": {
        "os",
        "as",
        "de",
        "que",
        "do",
        "da",
        "em",
        "um",
        "uma",
        "para",
        "com",
        "não",
        "se",
        "por",
        "no",
        "na",
        "dos",
        "mais",
        "como",
        "são",
        "pelo",
        "sua",
        "seu",
        "das",
    },
    "nl": {
        "de",
        "het",
        "een",
        "van",
        "en",
        "in",
        "is",
        "dat",
        "op",
        "te",
        "met",
        "voor",
        "zijn",
        "aan",
        "niet",
        "door",
        "ook",
        "als",
        "bij",
        "om",
        "worden",
    },
}

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

#: The winner must beat the runner-up by this much of the token stream, else the
#: evidence is too thin to name a language (Italian/Spanish/Portuguese overlap).
_MARGIN = 0.01


def detect_language(text: str, min_words: int = 40) -> str | None:
    """Best-guess ISO code for Latin-script text, or None when unsure."""
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if len(words) < min_words:
        return None

    scores = {
        lang: sum(1 for w in words if w in vocab) / len(words) for lang, vocab in _STOPWORDS.items()
    }
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best, runner_up = ranked[0], ranked[1]
    if best[1] < 0.04 or best[1] - runner_up[1] < _MARGIN:
        return None
    return best[0]


ModelScope = Literal["english", "multilingual", "unknown"]

# Checked in order; a name matching a multilingual hint is never English-only.
_MULTILINGUAL_HINTS = (
    "multilingual",
    "labse",
    "bge-m3",
    "xlm",
    "jina-embeddings-v3",
    "-m3",
)
_ENGLISH_HINTS = (
    "-en-",
    "-en",
    "_en",
    "all-minilm",
    "all-mpnet",
    "gte-small",
    "gte-base",
    "gte-large",
    "nomic-embed-text",
)


def model_language_scope(model_name: str) -> ModelScope:
    """Whether a model name identifies it as English-only or multilingual.

    Name-based, so it returns "unknown" rather than guess for models it does not
    recognise: a wrong warning is worse than no warning.
    """
    name = model_name.lower()
    if any(hint in name for hint in _MULTILINGUAL_HINTS):
        return "multilingual"
    if any(hint in name for hint in _ENGLISH_HINTS):
        return "english"
    return "unknown"


#: Suggested when an English-only model meets a corpus that is not English.
MULTILINGUAL_SUGGESTION = "intfloat/multilingual-e5-small"
