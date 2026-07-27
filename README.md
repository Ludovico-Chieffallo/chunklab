# chunklab

> **Find out which chunking strategy actually retrieves your answers best — on your own documents — in 60 seconds.**

[![CI](https://github.com/Ludovico-Chieffallo/chunklab/actions/workflows/ci.yml/badge.svg)](https://github.com/Ludovico-Chieffallo/chunklab/actions)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

When a RAG system gives a wrong answer, the failure is often in **retrieval** — the passage that holds the answer never reaches the model — not in the LLM's reasoning over what it was given. And one of the most overlooked variables in retrieval is **chunking**: how you split documents before embedding them. Independent benchmarks that hold the embedding model and retriever fixed and vary *only* the chunking find that it moves retrieval quality on the same corpus (see, e.g., Chroma's [Evaluating Chunking Strategies for Retrieval](https://www.trychroma.com/research/evaluating-chunking)).

On the example corpus bundled with this repo, the choice of strategy moves per-document recall@5 by up to **28 points** (API-reference document: 0.84 best vs 0.56 worst strategy), and **three different strategies win depending on the document** — reproduce both claims with `chunklab run --docs examples/corpus --questions examples/questions.yaml` (enforced in CI by `tests/test_corpus_discriminates.py`).

There is no universal best strategy: which one wins depends on **your** documents and **your** questions. `chunklab` makes that a one-command, fully-local, evidence-based decision instead of a guess.

## What it does

You give it your documents and a handful of questions (each tagged with the "gold" passage that answers it). It runs several chunking strategies, indexes and retrieves for each, and tells you **which strategy actually retrieves your answer-bearing text best — and why**.

```
docs + questions ─▶ [fixed · recursive · semantic · structure] ─▶ ranked report + diagnostics
```

- **Runs fully locally.** Default embeddings are a small local model (`BAAI/bge-small-en-v1.5`) — no API key, your documents never leave your machine.
- **Explains itself.** Per-strategy diagnostics (token distribution, % tiny fragments, boundary health, table integrity) tell you *why* a strategy won or lost.
- **Three outputs:** a console table, a standalone HTML report (per-question drill-down + chunk-boundary visualization), and a machine-readable JSON report for CI.

## Install

```bash
pip install chunklab
```

The first run downloads the embedding model (~130 MB) once and caches it.

## 60-second quickstart

```bash
# clone the repo to get the example corpus, or point --docs at your own files
chunklab run --docs examples/corpus --questions examples/questions.yaml
```

Console output of exactly that command (regenerated from a real run, never hand-edited):

<!-- BEGIN GENERATED EXAMPLE (scripts/regen_readme_example.py) -->
```
ChunkLab — 5 document(s), 129 scored questions, top_k=5, model=BAAI/bge-small-en-v1.5

 Strategy      ┃ balanced ┃ recall@5 ┃  MRR ┃ prec@5 ┃ tok@5 ┃ #chunks ┃ med_tok ┃ %tiny ┃ boundary 
━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━
 ▶ structure   │     0.81 │     0.81 │ 0.63 │   0.17 │  2075 │      71 │     346 │   25% │     100% 
   fixed       │     0.78 │     0.79 │ 0.61 │   0.18 │  2431 │      61 │     512 │    2% │       2% 
   recursive   │     0.77 │     0.78 │ 0.59 │   0.16 │  2237 │      62 │     458 │    5% │     100% 
   semantic    │     0.69 │     0.72 │ 0.55 │   0.15 │  3142 │      47 │     592 │    2% │     100% 
   semantic_n… │     0.68 │     0.70 │ 0.53 │   0.14 │  2886 │      59 │     470 │   24% │     100% 
recall/MRR/prec: retrieval quality at k=5 · tok@5: mean tokens retrieved per question (context cost)
· %tiny: chunks under the size floor · boundary: chunks not cut mid-sentence · full definitions: 
docs/metrics.md

Recommendation:
  No winner: 'structure' and 'fixed' are statistically indistinguishable on 129 scored questions 
(recall difference +0.019, 95% CI [-0.062, +0.101] includes zero). Roughly 2276 scored questions 
would be needed to separate them at the observed difference. Add questions before committing to a 
strategy.
```
<!-- END GENERATED EXAMPLE -->

Note what the recommendation does here: the top two strategies are within noise of each other on 129 questions, so chunklab **refuses to name a winner** and tells you how many questions would settle it. A tool that always produces a confident answer is the problem this one exists to fix.

The run is still decisive where the data supports it: `structure`, `fixed` and `recursive` are statistically tied on this corpus, but both `semantic` variants are *separated* from them (paired-bootstrap 95% CI of the recall difference excludes zero: +0.092 and +0.115). So the actionable output is "don't ship semantic chunking on this corpus, and pick among the top three on context cost" — which is exactly what the `balanced` ranking does, preferring `structure` at 2,075 retrieved tokens over `fixed` at 2,431.

Open `chunklab_report/report.html` for the full drill-down and the chunk-boundary visualization. The example corpus itself is documented in [`examples/CORPUS.md`](examples/CORPUS.md) — five documents designed so that different strategies win on different documents.

## Writing your `questions.yaml`

The tool scores retrieval offline by checking whether a **gold snippet** — a verbatim (or near-verbatim) passage from your document that answers the question — lands inside a retrieved chunk. Aim for 10–20 questions.

```yaml
questions:
  - id: q1
    query: "What is the termination notice period?"
    gold_snippets:
      - "written notice at least 30 days prior to termination"
    tags: [contracts]
  - id: q2
    query: "How is overtime compensated?"
    gold_snippets:
      - "Overtime is paid at 1.5x the regular hourly rate"
  - id: q3
    query: "What is the dress code?"
    # no gold_snippets -> skipped (with a warning), so scoring stays deterministic
```

Tips:
- Copy the gold snippet **verbatim** from the source so matching is reliable (small drift is absorbed by fuzzy matching, threshold configurable).
- A question with no `gold_snippets` is skipped — add the passage to include it.
- Aim for 15–30 questions: below 15, differences between strategies are usually noise, and chunklab will say so rather than pick a winner.

### Starting from a blank page

```bash
chunklab bootstrap --docs ./docs --out questions.draft.yaml -n 20
```

Drafts a question set locally (no API key): gold snippets are verbatim sentences that state a fact, queries are mechanical drafts marked `reviewed: false`. Rewrite the queries in your users' words before trusting the results — `chunklab run` keeps warning until you do.

### Check the question set before running

```bash
chunklab validate --docs ./docs --questions questions.yaml
```

`validate` catches the mistakes that silently ruin an evaluation — a snippet that drifted from the source scores zero for every strategy and looks like a chunking problem. For each one it prints the **verbatim source text, ready to paste**, and it exits non-zero so it can gate CI:

```
ERROR q1 (not_found): gold snippet not found in the corpus (closest match 81%):
'300 requests per minute on the Starter plan'
      found at api_reference:2492, verbatim source text:
      '300 requests per minute on Starter, 1,200 on'
```

The full workflow, honestly timed, is in [`docs/getting-started.md`](docs/getting-started.md).

## Configuration

`chunklab run` works with no config. To customize, pass `--config config.yaml`:

```yaml
embedding:
  backend: local                 # local (default) | openai (coming soon)
  model: BAAI/bge-small-en-v1.5
retrieval:
  mode: dense                    # dense (bm25/hybrid coming soon)
  top_k: 5
eval:
  fuzzy_threshold: 0.90
  ranking_metric: balanced       # balanced (default) | recall_at_k | mrr | hit_rate_at_k
  balanced_lambda: 0.05          # how hard to penalize context cost
  min_floor_tokens: 200
strategies:
  - { name: fixed,     params: { chunk_size: 512, overlap: 64 } }
  - { name: recursive, params: { chunk_size: 512, overlap: 64 } }
  - { name: semantic,  params: { breakpoint_percentile: 95, min_tokens: 200, max_tokens: 1000 } }
  - { name: structure, params: { max_tokens: 800 } }
output:
  formats: [console, html, json]
  dir: ./chunklab_report
```

See [`examples/config.example.yaml`](examples/config.example.yaml) for the full default.

### The chunking strategies

| Strategy | What it does |
|---|---|
| `fixed` | Fixed-size token windows with overlap. |
| `recursive` | Recursive split on paragraph → sentence → word separators. |
| `semantic` | Embedding-based boundaries **with a minimum-size floor** that merges tiny fragments — the fix for the "fragment trap." |
| `semantic_no_floor` | The naive version, included on purpose to show what the floor prevents. |
| `structure` | Heading-aware: one chunk per section, sub-split only when oversized. |

Run `chunklab strategies` to list them.

## Python API

```python
from chunklab import evaluate

report = evaluate(
    docs="./docs",
    questions="./questions.yaml",
    config=None,  # or a path / a Config object
)
print(report.recommendation)
for r in report.strategy_results:      # ranked best-first
    print(r.strategy, r.recall_at_k, r.mrr)
```

The `EvalReport` schema is a versioned public contract (`report.schema_version`, documented field-by-field in [`docs/schema.md`](docs/schema.md)) — serialize it with `report.model_dump_json()` for CI or dashboards.

## Web demo

```bash
pip install "chunklab[demo]"
chunklab demo          # launches a local Gradio app
```

Upload a document, type a few questions with gold snippets, pick strategies, and see the ranked comparison plus the chunk visualization.

## What this is (and isn't)

**It is** a fast, focused, local pre-flight utility that answers exactly one question: *which chunking strategy retrieves best on my corpus, and why?*

**It is not** a production RAG framework, a vector database, a document parser, or a general LLM-eval platform. It deliberately does one thing well.

## Supported inputs

Documents: **PDF, DOCX, TXT, MD**. Corpora of tens to low-hundreds of documents (this is a pre-flight tool, not a batch pipeline).

## License

[MIT](LICENSE) — permissive on purpose. Runs locally, no telemetry, no phone-home.
