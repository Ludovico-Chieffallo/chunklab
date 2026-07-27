"""Heuristic question-set bootstrapping (roadmap 3.2).

The point is not to produce a finished eval set — no heuristic can. It is to
replace a blank page with a reviewable draft: every gold snippet is a verbatim
sentence from the corpus, and every generated query is a mechanical
transformation of that sentence, marked `reviewed: false` until a human has
looked at it.

Selection favors sentences that state a fact worth asking about (a quantity, a
duration, an amount) and that fit a subject-auxiliary-predicate template, so
the generated query reads like a question rather than a fill-in-the-blank.
"""

import math
import re
from collections import Counter

from chunklab.models import Document, Question
from chunklab.text_utils import count_tokens, sentence_spans

# Auxiliaries/copulas the template can pivot on, in priority order.
_AUX = [
    "must be",
    "may be",
    "can be",
    "will be",
    "shall be",
    "has to be",
    "is",
    "are",
    "was",
    "were",
    "must",
    "may",
    "shall",
    "can",
    "will",
]

# Longest-first: Python alternation is leftmost-match, so "seven" before
# "seventy" would truncate "seventy-two".
_NUMBER_WORD_LIST = sorted(
    [
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "eighteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
        "thousand",
        "million",
    ],
    key=len,
    reverse=True,
)
_NUMBER_WORDS = "|".join(_NUMBER_WORD_LIST)

# Unit families are kept apart because each one decides a different wh-word.
_PERCENT_UNITS = r"percent|per\s+cent"
_DURATION_UNITS = r"business\s+days?|days?|hours?|minutes?|seconds?|weeks?|months?|years?"
_RATE_UNITS = r"[kmgt]bps|[kmgt]b/s|iops|rps|qps"
_SIZE_UNITS = r"[kmgt]ib|[kmgt]b|bytes?"
_MONEY_UNITS = r"euros?|dollars?"
_COUNT_UNITS = (
    r"requests?|calls?|users?|people|employees|documents?|records?|rows?|files?|items?|times"
)

# Rates before sizes: alternation is leftmost-first, and "Gbps" opens with "Gb".
_UNITS = "|".join(
    (_PERCENT_UNITS, _DURATION_UNITS, _RATE_UNITS, _SIZE_UNITS, _MONEY_UNITS, _COUNT_UNITS)
)
# A unit must not be the prefix of a longer word: without this, "5 Gbps" matched as
# "5 Gb" and the question came out as "How many is throughput capped?".
_UNIT_END = r"(?![A-Za-z])"

# A "focus" is the fact the question will ask for. Contracts spell numbers out and
# repeat them as numerals — "thirty (30) days" — so the parenthetical is optional.
_FOCUS_RE = re.compile(
    rf"\b(?:\d[\d,.]*\s*%|\d[\d,.]*x|[$€£]\s?\d[\d,.]*|\d[\d,.]*"
    rf"|(?:{_NUMBER_WORDS})(?:[-\s](?:{_NUMBER_WORDS}))?)"
    rf"(?:\s*\(\d[\d,.]*\))?"
    rf"(?:\s+(?:{_UNITS}){_UNIT_END})?",
    re.IGNORECASE,
)

_DURATION = re.compile(rf"\b(?:{_DURATION_UNITS}){_UNIT_END}", re.IGNORECASE)
_PERCENT = re.compile(rf"%|\b(?:{_PERCENT_UNITS}){_UNIT_END}", re.IGNORECASE)
_MONEY = re.compile(rf"[$€£]|\b(?:{_MONEY_UNITS}){_UNIT_END}", re.IGNORECASE)
_RATE = re.compile(rf"\b(?:{_RATE_UNITS}){_UNIT_END}", re.IGNORECASE)
_SIZE = re.compile(rf"\b(?:{_SIZE_UNITS}){_UNIT_END}", re.IGNORECASE)
_COUNT = re.compile(rf"\b(?:{_COUNT_UNITS}){_UNIT_END}", re.IGNORECASE)

# Function words that read badly at the end of a truncated question.
_TRAILING_NOISE = {
    "at",
    "of",
    "to",
    "in",
    "by",
    "with",
    "for",
    "from",
    "on",
    "per",
    "up",
    "within",
    "after",
    "before",
    "least",
    "most",
    "than",
    "and",
    "or",
    "the",
    "a",
    "an",
    "into",
    "over",
    "under",
    "about",
    "between",
    "as",
    "that",
    "which",
    "no",
    "more",
    "only",
    "every",
    "each",
    "any",
    "all",
    "its",
    "their",
    "his",
    "her",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")


UNTYPED = "What"

_YEAR = re.compile(r"\b(19|20)\d{2}\b")
# A number glued to an identifier (RFC 3339, HTTP 200, ISO 27001, NIST 800-88).
_IDENTIFIER_CONTEXT = re.compile(r"\b[A-Z]{2,}[-\s]?$")


def _wh_word(focus: str) -> str:
    if _PERCENT.search(focus):
        return "What percentage"
    if _DURATION.search(focus):
        return "How long"
    if _MONEY.search(focus):
        return "How much"
    # Throughput and data volumes are mass quantities: "how much", never "how many".
    if _RATE.search(focus) or _SIZE.search(focus):
        return "How much"
    if _YEAR.fullmatch(focus.strip()):
        return "When"
    if _COUNT.search(focus):
        return "How many"
    return UNTYPED  # bare number: no wh-word fits reliably


def _find_aux(sentence: str) -> tuple[str, int, int] | None:
    """First auxiliary in the sentence: (text, start, end) char offsets."""
    lowered = sentence.lower()
    best: tuple[str, int, int] | None = None
    for aux in _AUX:
        m = re.search(rf"\b{re.escape(aux)}\b", lowered)
        if m and (best is None or m.start() < best[1]):
            best = (sentence[m.start() : m.end()], m.start(), m.end())
    return best


def _cut_at_parenthetical(words: list[str]) -> list[str]:
    """Stop the predicate at the first comma: what follows is an aside.

    "capped, in each calendar month, at thirty percent" asks better as
    "capped" than as the whole run-on.
    """
    for i, word in enumerate(words):
        if word.endswith((",", ";")):
            return words[: i + 1]
    return words


def _trim_trailing_noise(words: list[str]) -> list[str]:
    while words and words[-1].lower().strip(",;:") in _TRAILING_NOISE:
        words.pop()
    return words


#: Sentences opening with these break the subject-auxiliary template.
_BAD_OPENERS = {
    "with",
    "when",
    "if",
    "where",
    "although",
    "because",
    "after",
    "before",
    "while",
    "unless",
    "since",
    "as",
    "for",
    "in",
    "on",
    "at",
    "by",
    "during",
    "under",
    "this",
    "that",
    "these",
    "those",
    "it",
    "there",
    "such",
    "both",
    "either",
    "neither",
}

MAX_PREDICATE_WORDS = 10

#: A subordinator between the verb and the focus means the fact belongs to an inner
#: clause, so the question would attach it to the wrong subject: "What percentage is
#: class imbalance handled automatically when the minority class falls below?"
_SUBORDINATORS = {
    "when",
    "whenever",
    "if",
    "while",
    "unless",
    "because",
    "where",
    "although",
    "though",
    "since",
    "after",
    "before",
    "until",
    "provided",
    "whereas",
    "whether",
    "that",
    "which",
    "who",
    "whose",
}

#: A comparative left at the end of a trimmed predicate has lost its complement:
#: "the greater of five percent and ..." becomes a dangling "the greater".
_DANGLING_COMPARATIVES = {
    "greater",
    "lesser",
    "larger",
    "smaller",
    "higher",
    "lower",
    "longer",
    "shorter",
    "later",
    "earlier",
    "fewer",
    "sooner",
}


def _focus_candidates(sentence: str):
    """Focus matches that are facts, not parts of identifiers like 'RFC 3339'."""
    for m in _FOCUS_RE.finditer(sentence):
        before = sentence[: m.start()].rstrip()
        if _IDENTIFIER_CONTEXT.search(before + " "):
            continue
        yield m


def build_query(sentence: str) -> str | None:
    """Turn a factual sentence into a draft question, or None if it doesn't fit."""
    sentence = " ".join(sentence.split())
    if "**" in sentence:
        return None  # FAQ heading line: it is already a question
    first_word = sentence.split()[0].lower().strip(",;:") if sentence.split() else ""
    if first_word in _BAD_OPENERS:
        return None

    aux = _find_aux(sentence)
    if not aux:
        return None
    focus_match = next((m for m in _focus_candidates(sentence) if m.start() > aux[1]), None)
    if not focus_match:
        return None  # no fact after the verb: the template would garble it

    aux_text, aux_start, aux_end = aux
    subject = sentence[:aux_start].strip().strip(",;:")
    predicate_words = sentence[aux_end : focus_match.start()].split()
    if not subject or len(subject.split()) > 8:
        return None
    if len(predicate_words) > MAX_PREDICATE_WORDS:
        return None  # too far from verb to focus: the question would ramble
    if any(w.lower().strip(",;:") in _SUBORDINATORS for w in predicate_words):
        return None  # the focus sits in an inner clause, not in the main one

    predicate_words = _cut_at_parenthetical(predicate_words)
    # _trim_trailing_noise pops in place, so keep a copy to see what it removed.
    kept = _trim_trailing_noise(list(predicate_words))
    dropped = tuple(w.lower().strip(",;:") for w in predicate_words[len(kept) :])
    predicate = " ".join(kept).rstrip(",;:")
    if not predicate or _PARTIAL_TAIL.search(predicate):
        return None  # trailing "per-", "a /", ... would read as a broken question
    if predicate.split()[-1].lower() in _DANGLING_COMPARATIVES:
        return None  # "not exceed the greater" lost the complement it compares to

    # A question reads better with a lowercase subject; acronyms keep their case.
    # The article stays: dropping it turned "A report of a production outage" into
    # the ungrammatical "report of a production outage".
    head = subject.split()[0]
    if not (len(head) > 1 and head.isupper()):  # "A report" is an article, not an acronym
        subject = subject[0].lower() + subject[1:]

    wh = _wh_word(focus_match.group(0))
    if wh == UNTYPED:
        return None  # a bare number rarely yields an answerable question

    # "capped at 30%" asks as "capped at?", not "capped?". Strand only a single
    # bare preposition: "payable within?" or "granted up to?" read worse than the
    # plain form, and "at" only fits an amount.
    if dropped == ("to",) or (dropped == ("at",) and wh in {"How much", "What percentage"}):
        predicate += f" {dropped[0]}"

    # A compound auxiliary inverts around the subject: "can custom images be
    # imported", not "can be custom images imported".
    modal, _, rest = aux_text.lower().partition(" ")
    inverted = f"{modal} {subject} {rest}" if rest else f"{modal} {subject}"

    query = f"{wh} {inverted} {predicate}?"
    if len(query.split()) > MAX_QUERY_WORDS:
        return None
    return query


#: A predicate ending mid-token ("per-", "a /") produces a broken question.
_PARTIAL_TAIL = re.compile(r"(?:[-/]|\b[a-z])$")
MAX_QUERY_WORDS = 16


def _idf(documents: list[Document]) -> dict[str, float]:
    doc_freq: Counter[str] = Counter()
    for d in documents:
        doc_freq.update({w.lower() for w in _WORD_RE.findall(d.text)})
    n = max(len(documents), 1)
    return {w: math.log(n / (1 + f)) + 1.0 for w, f in doc_freq.items()}


def _information_score(sentence: str, idf: dict[str, float]) -> float:
    words = [w.lower() for w in _WORD_RE.findall(sentence)]
    if not words:
        return 0.0
    rare = sum(idf.get(w, 1.0) for w in words) / len(words)
    focus_bonus = 1.5 if _FOCUS_RE.search(sentence) else 0.0
    return rare + focus_bonus


def generate_questions(
    documents: list[Document],
    n: int = 20,
    min_tokens: int = 10,
    max_tokens: int = 60,
    max_per_document: int | None = None,
) -> list[Question]:
    """Draft `n` questions with verbatim gold snippets, spread across documents."""
    idf = _idf(documents)
    if max_per_document is None:
        max_per_document = max(2, math.ceil(n / max(len(documents), 1)) + 1)

    candidates: list[tuple[float, str, Question]] = []
    for doc in documents:
        spans = sentence_spans(doc.text)
        seen_positions: list[int] = []
        doc_candidates: list[tuple[float, int, Question]] = []
        for start, end in spans:
            sentence = doc.text[start:end]
            flat = " ".join(sentence.split())
            if not (min_tokens <= count_tokens(flat) <= max_tokens):
                continue
            if flat.startswith("#") or flat.startswith("|"):
                continue  # headings and table rows need their own templates
            query = build_query(flat)
            if not query:
                continue
            doc_candidates.append(
                (
                    _information_score(flat, idf),
                    start,
                    Question(
                        id="",  # assigned after selection
                        query=query,
                        gold_snippets=[flat],
                        tags=[f"src:{doc.id}", "generated"],
                        reviewed=False,
                    ),
                )
            )

        doc_candidates.sort(key=lambda c: -c[0])
        taken = 0
        for score, start, question in doc_candidates:
            if taken >= max_per_document:
                break
            # Spread out: skip sentences adjacent to one already taken.
            if any(abs(start - p) < 400 for p in seen_positions):
                continue
            seen_positions.append(start)
            candidates.append((score, doc.id, question))
            taken += 1

    candidates.sort(key=lambda c: -c[0])
    selected = candidates[:n]
    for i, (_, _, question) in enumerate(selected, 1):
        question.id = f"gen{i:02d}"
    return [q for _, _, q in selected]


def dump_questions_yaml(questions: list[Question]) -> str:
    """Serialize a draft question set, with provenance comments."""
    lines = [
        "# Generated by `chunklab bootstrap` (heuristic backend).",
        "# Every gold snippet is verbatim source text; the queries are mechanical",
        "# drafts. Review and rewrite each query, then set reviewed: true.",
        "questions:",
    ]
    for q in questions:
        src = next((t.removeprefix("src:") for t in q.tags if t.startswith("src:")), "?")
        lines.append(f"  # from {src}")
        lines.append(f"  - id: {q.id}")
        lines.append(f"    query: {_yaml_str(q.query)}")
        lines.append("    gold_snippets:")
        for gold in q.gold_snippets:
            lines.append(f"      - {_yaml_str(gold)}")
        lines.append(f"    tags: [{', '.join(q.tags)}]")
        lines.append("    reviewed: false")
    return "\n".join(lines) + "\n"


def _yaml_str(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
