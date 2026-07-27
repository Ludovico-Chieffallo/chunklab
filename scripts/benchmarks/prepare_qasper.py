"""Convert QASPER into a chunklab corpus + question set.

QASPER (Dasigi et al., NAACL 2021, CC BY 4.0) is the benchmark chunklab needs
most, because of *how* it was built: questions were written by NLP practitioners
who had read only the paper's title and abstract, and the supporting evidence
was then annotated by different people who read the full paper. Neither the
questions nor the gold spans come from whoever wrote the documents — which is
exactly the independence chunklab's own example corpus cannot have.

Conversion choices, all of which affect the numbers and are therefore reported:

* One annotator's evidence per question (the first answer that cites any),
  rather than the union across annotators. Annotators disagree about which
  paragraphs support an answer; unioning their sets would inflate `gold_total`
  with disagreement and make recall look worse for reasons that have nothing to
  do with chunking.
* Evidence referring to figures and tables ("FLOAT SELECTED: ...") is dropped:
  those captions are not part of the rendered document text, so no chunker could
  ever retrieve them.
* Unanswerable questions are dropped - there is nothing to retrieve.
* Any remaining evidence string that is not a verbatim substring of the rendered
  document is dropped, and counted, rather than fuzzily matched.
* Evidence shorter than `--min-gold-tokens` is dropped. Running `chunklab
  validate` over the converted set found annotations whose entire evidence is the
  section heading "Experimental Setup" - a string that occurs in 6 of 30 sampled
  papers, so any chunk containing that heading would score as a hit and inflate
  every strategy equally. Pass `--min-gold-tokens 0` to reproduce the unfiltered
  set.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import (  # noqa: E402
    DEFAULT_CACHE,
    ConversionStats,
    download,
    write_questions_yaml,
)

from chunklab.text_utils import count_tokens  # noqa: E402

QASPER_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
DEV_FILE = "qasper-dev-v0.3.json"

#: Evidence entries that point at an image caption rather than body text.
_FLOAT_PREFIX = "FLOAT SELECTED"


def fetch_dev_set(cache: Path) -> dict:
    archive = download(QASPER_URL, cache / "qasper-train-dev-v0.3.tgz")
    extracted = cache / DEV_FILE
    if not extracted.exists():
        with tarfile.open(archive) as tar:
            member = tar.getmember(DEV_FILE)
            with tar.extractfile(member) as source:
                extracted.write_bytes(source.read())
    return json.loads(extracted.read_text(encoding="utf-8"))


def render_paper(paper: dict) -> str:
    """Render a paper as markdown, preserving paragraph text verbatim."""
    parts = [f"# {paper['title'].strip()}", "", "## Abstract", "", paper["abstract"].strip()]
    for section in paper["full_text"]:
        name = (section.get("section_name") or "").strip()
        if name:
            parts += ["", f"## {name}"]
        for paragraph in section["paragraphs"]:
            text = paragraph.strip()
            if text:
                parts += ["", text]
    return "\n".join(parts) + "\n"


def _first_evidence(qa: dict) -> list[str]:
    """Evidence from the first annotator who cited any body-text paragraph."""
    for answer_block in qa.get("answers", []):
        answer = answer_block.get("answer", {})
        if answer.get("unanswerable"):
            continue
        evidence = [
            span.strip()
            for span in answer.get("evidence", [])
            if span.strip() and not span.strip().startswith(_FLOAT_PREFIX)
        ]
        if evidence:
            return evidence
    return []


def _safe_id(arxiv_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", arxiv_id)


def convert(
    papers: dict,
    paper_ids: list[str],
    stats: ConversionStats,
    min_gold_tokens: int = 5,
) -> tuple[dict, list[dict]]:
    documents: dict[str, str] = {}
    questions: list[dict] = []

    for arxiv_id in paper_ids:
        paper = papers[arxiv_id]
        doc_id = _safe_id(arxiv_id)
        text = render_paper(paper)
        documents[doc_id] = text
        stats.documents += 1

        for index, qa in enumerate(paper.get("qas", [])):
            stats.questions_seen += 1
            evidence = _first_evidence(qa)
            if not evidence:
                stats.drop("unanswerable, or evidence is only a figure/table caption")
                continue

            verbatim = [span for span in evidence if span in text]
            if len(verbatim) != len(evidence):
                stats.drop(
                    "evidence span not found verbatim in the rendered paper",
                    len(evidence) - len(verbatim),
                )

            long_enough = [span for span in verbatim if count_tokens(span) >= min_gold_tokens]
            if len(long_enough) != len(verbatim):
                stats.drop(
                    f"evidence shorter than {min_gold_tokens} tokens (matches by accident)",
                    len(verbatim) - len(long_enough),
                )
            verbatim = long_enough
            if not verbatim:
                stats.drop("no evidence span survived filtering")
                continue

            questions.append(
                {
                    "id": f"{doc_id}_q{index:02d}",
                    "query": " ".join(qa["question"].split()),
                    "gold_snippets": verbatim,
                    "tags": [f"src:{doc_id}", "qasper"],
                }
            )
            stats.questions_kept += 1
            stats.gold_snippets += len(verbatim)

    return documents, questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--papers", type=int, default=30, help="How many papers to sample.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--min-gold-tokens",
        type=int,
        default=5,
        help="Drop evidence shorter than this; 0 reproduces the unfiltered set.",
    )
    args = parser.parse_args()

    papers = fetch_dev_set(args.cache)
    # Sorted before sampling so the selection depends only on the seed, never on
    # dict iteration order.
    all_ids = sorted(papers)
    sampled = random.Random(args.seed).sample(all_ids, min(args.papers, len(all_ids)))

    stats = ConversionStats()
    documents, questions = convert(papers, sorted(sampled), stats, args.min_gold_tokens)

    corpus_dir = args.out / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, text in documents.items():
        (corpus_dir / f"{doc_id}.md").write_text(text, encoding="utf-8")

    write_questions_yaml(
        args.out / "questions.yaml",
        questions,
        [
            "# QASPER (Dasigi et al., NAACL 2021), CC BY 4.0.",
            f"# Generated by scripts/benchmarks/prepare_qasper.py --papers {args.papers}"
            f" --seed {args.seed} --min-gold-tokens {args.min_gold_tokens}",
            "# Questions were written by NLP practitioners who had seen only the abstract;",
            "# evidence spans were annotated separately by readers of the full paper.",
        ],
    )

    print(stats.render())
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
