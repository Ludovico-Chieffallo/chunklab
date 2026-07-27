"""Token counting and sentence splitting with char spans."""

import re
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
    offsets back to char offsets.
    """
    enc = _encoder()
    tokens = enc.encode(text, disallowed_special=())
    # byte offset -> char offset map
    byte_to_char: dict[int, int] = {}
    b = 0
    for i, ch in enumerate(text):
        byte_to_char[b] = i
        b += len(ch.encode("utf-8"))
    byte_to_char[b] = len(text)

    spans: list[tuple[int, int]] = []
    pos = 0
    for tok in tokens:
        n = len(enc.decode_single_token_bytes(tok))
        start = byte_to_char.get(pos)
        end = byte_to_char.get(pos + n)
        # A token boundary can split a multi-byte char; extend to the nearest
        # char boundary on either side.
        if start is None:
            start = byte_to_char[max(k for k in byte_to_char if k < pos)]
        if end is None:
            end = byte_to_char[min(k for k in byte_to_char if k > pos + n)]
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
        before = text[: m.start() + 1]
        last_word = re.split(r"[\s(]", before.rstrip(".!?").rstrip())[-1].lower()
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
