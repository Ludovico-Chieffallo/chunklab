# chunklab

> **Tells you whether the difference between chunking strategies is real — on your own documents — before you commit to one.**

[![CI](https://github.com/Ludovico-Chieffallo/chunklab/actions/workflows/ci.yml/badge.svg)](https://github.com/Ludovico-Chieffallo/chunklab/actions)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

When a RAG system gives a wrong answer, the failure is often in **retrieval** — the passage that holds the answer never reaches the model — not in the LLM's reasoning over what it was given. And one of the most overlooked variables in retrieval is **chunking**: how you split documents before embedding them. Independent benchmarks that hold the embedding model and retriever fixed and vary *only* the chunking find that it moves retrieval quality on the same corpus (see, e.g., Chroma's [Evaluating Chunking Strategies for Retrieval](https://www.trychroma.com/research/evaluating-chunking)).

The trap is that chunking differences are usually **small and noisy**, and a few dozen questions cannot tell a real 3-point gap from a coin flip. Most comparisons declare a winner anyway. chunklab's job is to stop you from doing that.

Every recommendation it makes is gated by a bootstrap over your questions, and the bar is deliberately set where picking a winner is a *selection* rather than a coin flip: a strategy is named only when it comes out best in at least 90% of resamples of your question set. When nothing clears that bar it says so — and then says what the run *can* decide: which strategies are already ruled out, how many more questions would settle the rest (or that no realistic number would), and how much context each survivor costs you on every query, forever.

**On [public benchmarks](docs/benchmarks.md) this is not a hypothetical.** On QASPER (889 human-written questions, human-annotated evidence) the top two strategies differ by `+0.000` recall. At 70 questions one of them led by `+0.034` — noise that a less careful tool would have shipped as a recommendation. They are still not equivalent, though: one retrieves **35% fewer tokens** for the same recall, and that is the decision worth making.

There is no universal best strategy either: `recursive` ranks **first** on QASPER and **last** on CUAD. Which one wins depends on **your** documents and **your** questions — which is the whole reason to measure instead of guess.

## What it does

You give it your documents and a handful of questions (each tagged with the "gold" passage that answers it). It runs several chunking strategies, indexes and retrieves for each, and tells you **whether any of them is really better — and if not, which one is cheaper**.

```
docs + questions ─▶ [fixed · recursive · semantic · structure] ─▶ ranked report + diagnostics
```

- **Refuses to guess.** A bootstrap over your questions gates every recommendation: a strategy is named only when it is genuinely the best of the field in at least 90% of resamples, which prices in the fact that it was *chosen* from that field. A lead that could be noise is reported as an undecided run — naming what is still in play and what is already out — never as a winner.
- **Compares the retriever too.** `--compare-retrievers` evaluates every strategy under dense, BM25 and hybrid (RRF) retrieval. On the bundled corpus that axis moved recall more than the choice of chunker did — which one dominates is a property of *your* corpus, so it is measured rather than assumed.
- **Prices the tie.** When recall is indistinguishable, `tok@k` is the tiebreaker — the tokens each strategy spends on every query for the rest of the system's life.
- **Runs fully locally.** Default embeddings are a small local model (`BAAI/bge-small-en-v1.5`) — no API key, no telemetry, your documents never leave your machine.
- **Explains itself.** Per-strategy diagnostics (token distribution, % tiny fragments, boundary health, table integrity) tell you *why* a strategy won or lost.
- **Checks your questions first.** `chunklab validate` catches broken gold snippets before a run is spent on them — it found annotation artifacts in a *published* academic benchmark.
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
 ▶ recursive   │     0.82 │     0.82 │ 0.61 │   0.17 │  2228 │      62 │     458 │    5% │     100% 
   structure   │     0.81 │     0.81 │ 0.65 │   0.17 │  2128 │      71 │     346 │   25% │     100% 
   fixed       │     0.81 │     0.81 │ 0.62 │   0.19 │  2447 │      61 │     512 │    2% │       2% 
   semantic    │     0.72 │     0.73 │ 0.55 │   0.15 │  2862 │      49 │     593 │    0% │     100% 
   semantic_n… │     0.70 │     0.71 │ 0.53 │   0.15 │  2648 │      62 │     402 │   24% │     100% 
recall/MRR/prec: retrieval quality at k=5 · tok@5: mean tokens retrieved per question (context cost)
· %tiny: chunks under the size floor · boundary: chunks not cut mid-sentence · full definitions: 
docs/metrics.md
This ranking holds for BAAI/bge-small-en-v1.5. Strategy order changes with the embedding model — run
with the one you deploy.

Recommendation:
  No single winner: 'recursive' leads but is the best of the 5 strategies compared in only 42% of 
bootstrap resamples over 129 scored questions. 3 cannot be ruled out ('recursive', 'structure', 
'fixed'); 'semantic' (0%), 'semantic_no_floor' (0%) can. Roughly 6938 scored questions would be 
needed to separate the top two at the observed difference. Add questions before committing to a 
strategy.
```
<!-- END GENERATED EXAMPLE -->

Note what the recommendation does here. No strategy is best often enough across resamples to be named, so chunklab **refuses to name a winner** — but it does not stop at refusing. It reports which strategies are still in play, which the 129 questions already rule out, and how many more would settle the rest. A tool that always produces a confident answer is the problem this one exists to fix; a tool that only ever says "not enough data" is no better.

The run is still decisive where the data supports it. `recursive`, `structure` and `fixed` are statistically tied, but `semantic_no_floor` is *separated* from all three — recall differences +0.111, +0.099 and +0.103, every paired-bootstrap 95% CI excluding zero. The fragment trap it demonstrates is real and measurable, and the floored `semantic` variant recovers most of it. So the actionable output is "don't ship a semantic splitter without a minimum-size floor, then pick among the top three on context cost" — which is what the `balanced` ranking does, preferring `structure` at 2,128 retrieved tokens over `fixed` at 2,447 despite `fixed`'s marginally higher recall.

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

### When the answer lives in more than one place

On a corpus where several documents cover the same ground — review papers, product docs across versions, contracts from one template — a strategy can retrieve a perfectly good answer *from the wrong document* and score zero. Write the alternatives as a nested list and any one of them counts:

```yaml
  - id: q4
    query: "What DNA sequence does Cas9 need to recognise a target?"
    gold_snippets:
      # one slot, three acceptable passages
      - - "must be immediately adjacent to the NGG motif"
        - "the PAM is strictly required to be immediately next to the 3' end"
        - "the PAM is typically NGG"
```

Each top-level entry is a **slot** and recall is the fraction of slots filled, so alternatives raise the ceiling instead of the denominator. Plain strings still mean exactly what they always did — a slot with one acceptable passage — and mixing the two forms in one question is fine when some passages are genuinely required and others are interchangeable.

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
Gate a pipeline on retrieval regressions with `chunklab check` — [`docs/ci.md`](docs/ci.md).
Put your own splitter in the comparison: [`docs/extending.md`](docs/extending.md).
How the metrics are defined and why `balanced` is the default: [`docs/metrics.md`](docs/metrics.md).
Results on independent, human-annotated corpora, with every conversion choice and drop rate
reported: [`docs/benchmarks.md`](docs/benchmarks.md).

## Configuration

`chunklab run` works with no config. To customize, pass `--config config.yaml`:

```yaml
embedding:
  backend: local                 # local (default) | openai (coming soon)
  model: BAAI/bge-small-en-v1.5
retrieval:
  mode: dense                    # dense | bm25 | hybrid
  top_k: 5
  compare: []                    # e.g. [dense, bm25, hybrid] for a strategy x retriever matrix
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
for r in report.strategy_results:  # ranked best-first
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
