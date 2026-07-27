"""Token counting and sentence splitting with char spans."""

import re
from bisect import bisect_left, bisect_right
from functools import lru_cache


@lru_cache(maxsize=1)
def _encoder():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text, disallowed_special=()))


def token_spans(text: str) -> list[tuple[int, int]]:
    """Char span of every token in `text`, in order.

    tiktoken is byte-level, so we accumulate token byte lengths and map byte
    offsets back to char offsets. A token boundary can fall inside a multi-byte
    character, in which case the span extends to the enclosing char boundaries.

    `byte_at[i]` is the byte offset of char `i`, strictly increasing, with a
    final sentinel — so a boundary lookup is a binary search. Scanning the map
    instead made the function quadratic, which cost ~19 s on 48k chars of
    Japanese, where nearly every token boundary splits a character.
    """
    enc = _encoder()
    tokens = enc.encode(text, disallowed_special=())

    byte_at = [0] * (len(text) + 1)
    offset = 0
    for i, ch in enumerate(text):
        byte_at[i] = offset
        offset += len(ch.encode("utf-8"))
    byte_at[len(text)] = offset

    spans: list[tuple[int, int]] = []
    pos = 0
    for tok in tokens:
        n = len(enc.decode_single_token_bytes(tok))
        # last char boundary at or before pos, first at or after pos + n
        start = bisect_right(byte_at, pos) - 1
        end = bisect_left(byte_at, pos + n)
        spans.append((start, end))
        pos += n
    return spans


# Closing quotes/brackets and markdown emphasis (`**bold?**`) may sit between the
# terminal punctuation and the whitespace that ends the sentence.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])[\"')\]*_]*\s+|\n{2,}")

_ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "fig",
    "no",
    "vol",
    "inc",
    "ltd",
    "co",
    "dept",
    "approx",
}


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split text into sentences, returning (start, end) char spans.

    Regex-based (period/!/? followed by whitespace, or blank lines), with a
    small abbreviation list to reduce false splits. Spans cover the whole text
    minus leading/trailing whitespace of each sentence.
    """
    boundaries = [0]
    for m in _SENTENCE_END_RE.finditer(text):
        # Skip splits right after a known abbreviation like "Dr." or "e.g."
        # Whitespace first: `before` always ends with the whitespace that closed the
        # sentence, so stripping punctuation first was a no-op and left "dr." — which
        # never matched the list, making every abbreviation split a sentence.
        before = text[: m.start() + 1]
        last_word = re.split(r"[\s(]", before.rstrip().rstrip(".!?"))[-1].lower()
        if last_word in _ABBREVIATIONS:
            continue
        boundaries.append(m.end())
    boundaries.append(len(text))

    spans: list[tuple[int, int]] = []
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        seg = text[start:end]
        stripped = seg.strip()
        if not stripped:
            continue
        s = start + len(seg) - len(seg.lstrip())
        spans.append((s, s + len(stripped)))
    return spans
