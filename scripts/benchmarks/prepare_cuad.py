"""Convert CUAD into per-contract chunklab corpora.

CUAD (Hendrycks et al., NeurIPS 2021 Datasets & Benchmarks, CC BY 4.0) is 510
commercial contracts with 13k+ clause spans annotated by lawyers. It tests
something the example corpus cannot: whether a chunking strategy keeps a legal
clause intact and findable inside a long, densely formatted document.

**One contract per corpus, aggregated afterwards.** CUAD asks the same 41
category questions of every contract ("Highlight the parts related to Governing
Law..."). Pooling contracts into one corpus would make every question ambiguous
across all of them, and would measure "find a governing-law clause in any
contract" — a different task from the one CUAD annotates. Each contract is
therefore its own corpus and its own question set, and results are averaged over
contracts weighted by question count.

Conversion choices:

* Only questions with at least one annotated span are kept; CUAD marks absent
  categories `is_impossible`, and there is nothing to retrieve for those.
* Spans are used verbatim via their `answer_start` offset, so gold text always
  matches the contract exactly.
* Duplicate spans for one question are de-duplicated.
* Spans shorter than `--min-gold-tokens` are dropped: a two-word clause title
  matches by accident and inflates every strategy equally.
* The long "Details: ..." rubric is kept in the query. It is what CUAD actually
  asks, and shortening it would be a silent change to the benchmark.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import (  # noqa: E402
    DEFAULT_CACHE,
    ConversionStats,
    download,
    write_questions_yaml,
)

from chunklab.text_utils import count_tokens  # noqa: E402

CUAD_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
CUAD_MEMBER = "CUADv1.json"


def fetch_contracts(cache: Path) -> list[dict]:
    archive = download(CUAD_URL, cache / "cuad-data.zip")
    extracted = cache / CUAD_MEMBER
    if not extracted.exists():
        with zipfile.ZipFile(archive) as zf:
            member = next(n for n in zf.namelist() if n.endswith(CUAD_MEMBER))
            extracted.write_bytes(zf.read(member))
    return json.loads(extracted.read_text(encoding="utf-8"))["data"]


def _safe_id(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", title)[:80]


def convert_contract(
    contract: dict, stats: ConversionStats, min_gold_tokens: int
) -> tuple[str, list[dict]] | None:
    paragraph = contract["paragraphs"][0]
    context = paragraph["context"]
    questions: list[dict] = []

    for index, qa in enumerate(paragraph["qas"]):
        stats.questions_seen += 1
        if qa.get("is_impossible") or not qa.get("answers"):
            stats.drop("category absent from this contract (is_impossible)")
            continue

        spans: list[str] = []
        for answer in qa["answers"]:
            start = answer["answer_start"]
            text = context[start : start + len(answer["text"])]
            if text != answer["text"]:
                stats.drop("span offset did not match its text")
                continue
            if count_tokens(text) < min_gold_tokens:
                stats.drop(f"span shorter than {min_gold_tokens} tokens (matches by accident)")
                continue
            if text not in spans:
                spans.append(text)

        if not spans:
            stats.drop("no span survived filtering")
            continue

        questions.append(
            {
                "id": f"q{index:02d}",
                "query": " ".join(qa["question"].split()),
                "gold_snippets": spans,
                "tags": ["cuad"],
            }
        )
        stats.questions_kept += 1
        stats.gold_snippets += len(spans)

    if not questions:
        return None
    return context, questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contracts", type=int, default=15, help="How many contracts to sample.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--min-gold-tokens", type=int, default=5)
    args = parser.parse_args()

    contracts = fetch_contracts(args.cache)
    contracts.sort(key=lambda c: c["title"])
    sampled = random.Random(args.seed).sample(contracts, min(args.contracts, len(contracts)))

    stats = ConversionStats()
    written = 0
    for contract in sorted(sampled, key=lambda c: c["title"]):
        converted = convert_contract(contract, stats, args.min_gold_tokens)
        if converted is None:
            continue
        context, questions = converted
        doc_id = _safe_id(contract["title"])
        case_dir = args.out / doc_id
        (case_dir / "corpus").mkdir(parents=True, exist_ok=True)
        (case_dir / "corpus" / f"{doc_id}.md").write_text(context, encoding="utf-8")
        write_questions_yaml(
            case_dir / "questions.yaml",
            questions,
            [
                "# CUAD v1 (Hendrycks et al., 2021), CC BY 4.0.",
                f"# Contract: {contract['title']}",
                "# Clause spans annotated by lawyers; one contract per corpus by design.",
            ],
        )
        written += 1
        stats.documents += 1

    print(stats.render())
    print(f"\nwrote {written} contract corpora under {args.out}")


if __name__ == "__main__":
    main()
