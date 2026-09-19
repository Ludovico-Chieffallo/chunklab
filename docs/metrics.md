# Metrics reference

Every number chunklab reports, what it means, and — for the ranking metric — why it is
built the way it is.

## Retrieval metrics (per strategy, cutoff `k` = `retrieval.top_k`)

| Metric | Definition |
|---|---|
| `recall_at_k` | Mean over scored questions of (gold snippets found in the top-k retrieved chunks) / (total gold snippets of the question). |
| `hit_rate_at_k` | Fraction of questions with at least one gold snippet found in the top-k. |
| `mrr` | Mean reciprocal rank of the first hit; 0 when nothing hits within k. |
| `precision_at_k` | Mean fraction of the k retrieved chunks that contain a gold snippet. |
| `retrieved_tokens_at_k` | Mean total tokens of the k retrieved chunks per question — the context the downstream LLM must pay for and reason over. |
| `context_efficiency` | Mean (tokens of *found* gold snippets) / (tokens retrieved): how much of the retrieved context is actually answer-bearing. |

A "found" gold snippet means the snippet text is contained in a retrieved chunk (exact
containment after whitespace/case normalization, or fuzzy containment above
`eval.fuzzy_threshold`).

## The large-chunk bias, and the `balanced` ranking

Containment-based recall has a structural bias: **a bigger chunk always contains at least
as much gold as a smaller one**. A degenerate strategy that returns each document as a
single chunk maximizes `recall_at_k` while being useless in practice — the "answer" it
retrieves is an entire document the LLM must re-search, at full context price. Ranking
strategies purely by `recall_at_k` therefore silently rewards size.
`tests/test_degeneration.py` demonstrates the bias and pins the fix.

The **default** ranking metric, `balanced`, charges that context cost explicitly:

```
balanced(s) = recall_at_k(s) − λ · (retrieved_tokens_at_k(s) / T_min − 1)
```

where `T_min` is the smallest `retrieved_tokens_at_k` across the compared strategies and
`λ` = `eval.balanced_lambda` (default **0.05**).

Reading it: the leanest strategy pays no penalty; a strategy retrieving twice the minimum
pays λ (5 recall points at the default); ten times the minimum pays 9λ (45 points), which
no recall advantage can plausibly recover. The normalization on `T_min` makes the penalty
*relative to the field being compared*, so `balanced` never punishes a corpus for having
long documents — only a strategy for being fatter than its competitors.

Calibration rationale for λ = 0.05: among the non-degenerate default strategies the token
ratio `T/T_min` stays below ~2, so the penalty moves scores by at most ~5 points — enough
to break recall ties in favor of cheaper context, not enough to override a real recall
gap. Raising λ to 0.10 makes context cost dominate; lowering it to 0.02 makes `balanced`
nearly identical to `recall_at_k`. Set it per your own downstream token budget.

What this looks like on the example corpus (129 questions, `bge-small-en-v1.5`):

| strategy | recall@5 | tok@5 | balanced |
|---|---|---|---|
| `recursive` | 0.822 | 2228 | 0.819 |
| `structure` | 0.810 | 2128 | **0.810** |
| `fixed` | 0.814 | 2447 | 0.807 |

`fixed` retrieves marginally more than `structure` (+0.004 recall) while spending 15% more
context, and `balanced` ranks `structure` above it. It does **not** overturn `recursive`,
whose recall lead is larger than the penalty — which is the intended behaviour, and is
pinned by a slow test. On this corpus no recall gap is small enough for the penalty to
change first place; the metric breaks near-ties, it does not manufacture upsets.

## What recall measures, and what it does not

Recall is **span localisation**, not answerability: it asks whether the passage *you
annotated* reached the top k, not whether an answer did. On a corpus where one document
covers the ground, those are the same question. On a redundant corpus they are not.

Measured on 26 CRISPR review papers, 18 scored questions, `recursive` + hybrid at k=5.
Asked *"What DNA sequence requirement is necessary for Cas9 target recognition?"*, every
one of the 15 strategy × retriever cells retrieved a section headed `### PAM` reading
"The protospacer-adjacent motif (PAM) is strictly required to be immediately next to the
3' end of the target sequence… the PAM is typically NGG" — a fuller answer than the
annotated gold — and every one scored zero. Reading all seven missed questions by hand,
at least three were of this kind.

**chunklab does not try to detect this, and the reason is measured.** Two automatic
judges were built and rejected:

| judge | result on the same 90 (question, chunk) pairs |
|---|---|
| bi-encoder similarity to the gold | AUC **0.617** against chance 0.5; at a threshold calibrated on the chunks that did match, 73% of chunks that provably contain no gold were marked "equivalent" |
| `ms-marco-MiniLM-L-6-v2` cross-encoder | better — it rejected the passage about *adenine* editing that the bi-encoder scored 0.768 on a *cytosine* question — but still interleaved on the five hand-read cases |

A bi-encoder cannot separate "answers this" from "is about this", which is unsurprising:
it is the same model that did the retrieval, grading its own work. Shipping either would
mean a warning that fires on most misses and is right about half the time.

The fix is to let you declare the equivalence instead of having a model guess it. Write
the alternatives as a nested list and the slot is filled by any of them:

```yaml
gold_snippets:
  - - "must be immediately adjacent to the NGG motif"
    - "the PAM is strictly required to be immediately next to the 3' end"
    - "the PAM is typically NGG"
```

Before slots existed this made things *worse*: `gold_snippets` was conjunctive, so three
equally valid passages with one retrieved scored 0.33 and the honest annotation was
punished. Reproduce the improvement by adding the variants above to that question and
re-running: it goes from 0 of 2 slots to 1 of 1, and corpus recall from 0.639 to 0.694.

Until the alternatives are annotated, **read recall on a redundant corpus as a lower
bound**, and prefer comparing strategies to reading the absolute number.

## Retrieval mode

Chunking is only half of the pipeline. `--compare-retrievers` (or
`retrieval.compare: [dense, bm25, hybrid]`) evaluates every strategy under every
retriever and reports a **strategy × retriever matrix**, because a strategy ranked under
one retriever is a statement about that pairing, not about chunking in general.

- **`dense`** — cosine over the embedding model. Generalises across wording, blurs exact
  tokens.
- **`bm25`** — Okapi BM25, implemented in-tree (no dependency). Matches exact terms:
  identifiers, clause numbers, error codes, product names. Tokenization is word-based,
  falling back to **character bigrams** for scripts written without spaces (Han, Kana,
  Hangul, Thai), where a word tokenizer would emit one token per sentence.
- **`hybrid`** — Reciprocal Rank Fusion, `score = Σ 1/(60 + rank)`. Fuses *ranks*, not
  scores: BM25 scores are unbounded idf sums and cosines live in [-1, 1], so any weighted
  sum would need per-corpus calibration chunklab cannot do honestly. Ties are broken by
  best rank achieved and then chunk id, so the result does not depend on the order the
  retrievers were listed in.

`balanced` normalizes on the cheapest cell of the whole matrix, so with `compare` set the
context penalty prices retrievers against each other too, not just strategies.

### Reproducing the tables below

```bash
chunklab run --docs examples/corpus --questions examples/questions.yaml --compare-retrievers
```

The confidence intervals come from a paired bootstrap between the same strategy under two
retrievers, over the per-question recalls in `report.json`. The QASPER figures use the
corpus prepared by `scripts/benchmarks/prepare_qasper.py` (see
[public benchmarks](benchmarks.md)).

### What this measured, and a warning about it

On the bundled example corpus, switching from `dense` to `hybrid` improved recall@5 for
**every** strategy, by more than the spread between strategies:

| strategy | dense | bm25 | hybrid | hybrid − dense | 95% CI |
|---|---|---|---|---|---|
| `fixed` | 0.814 | 0.898 | 0.953 | **+0.140** | [+0.078, +0.209] |
| `semantic_no_floor` | 0.711 | 0.920 | 0.873 | **+0.163** | [+0.101, +0.225] |
| `structure` | 0.810 | 0.929 | 0.926 | **+0.116** | [+0.062, +0.178] |
| `semantic` | 0.734 | 0.912 | 0.858 | **+0.124** | [+0.062, +0.186] |
| `recursive` | 0.822 | 0.873 | 0.920 | **+0.098** | [+0.039, +0.160] |

**Do not generalise that table.** On QASPER, whose questions were written by people who
had read only the abstract, the same comparison is significant for **two strategies out of
five**: `semantic` (+0.131) and `semantic_no_floor` (+0.107); for `fixed`, `recursive` and
`structure` the interval includes zero.

The likely reason is a defect in the example corpus, not a property of BM25: its questions
and its documents were written by the same author, so the questions reuse the documents'
vocabulary — precisely what lexical matching rewards. Real users do not have that
advantage, and QASPER's protocol removes it by construction.

Which is the point of the matrix: whether the retriever or the chunker dominates is a
property of *your* corpus and *your* questions, and it is measurable rather than
guessable.

## Results are specific to your embedding model

Every ranking chunklab produces is a statement about *one* embedding model. That is not a
hedge — it is measured. Running the identical corpus and questions under a second model
changes the answer:

| | `bge-small-en-v1.5` | `all-MiniLM-L6-v2` |
|---|---|---|
| example corpus, winner | `recursive` (0.822) | **`fixed`** (0.721) |
| example corpus, order | recursive, structure, fixed, semantic, semantic_no_floor | fixed, recursive, structure, semantic_no_floor, semantic |
| QASPER, winner | `recursive` (0.402) | `recursive` (0.386) |
| QASPER, order | recursive, structure, semantic, fixed, semantic_no_floor | recursive, semantic, structure, semantic_no_floor, fixed |

Reproduce by setting `embedding.model` and re-running the command above.

The winner changed on one of the two corpora, and the ordering changed on both. So:

**Run chunklab with the embedding model you actually deploy.** Validating with the default
and shipping something else means acting on a conclusion that was never tested — the kind
of silent mismatch this tool exists to expose, and one it can commit itself if you let it.

If you have not chosen a model yet, this is also the honest reading: the model choice
interacts with the chunking choice, so pick the model first and compare strategies under
it, rather than treating the two as independent.

## Statistical honesty

With few questions, small metric differences are noise. chunklab therefore:

- reports `ci95` per strategy — a percentile bootstrap (default 10,000 resamples, seeded
  by `eval.seed`) confidence interval on mean per-question recall;
- gates the recommendation on the leader's **probability of actually being best** — the
  share of paired bootstrap resamples of your question set in which it comes out on top.
  A strategy is named only above `SELECTION_CONFIDENCE` (0.90); below it the report names
  no winner, lists the candidates that cannot be ruled out, names those that can, and
  estimates how many questions would settle the rest;
- warns when fewer than 15 scored questions are provided.

### Why a probability and not a corrected pairwise test

Sorting k candidates and then testing the top two is a **selection**, not a hypothesis
test. The maximum of k noisy estimates is biased upward, so an interval computed after
the field has been sorted is anti-conservative precisely when it is about to name a
winner. On a 15-cell run over 26 research papers the leading cell was genuinely best in
only 68% of resamples, while the report spoke as if the pair at the top were the only
candidates.

Two textbook repairs were measured on this project's own corpora before the current gate
was written, and both were rejected for making the tool *less* useful without making it
more true:

| repair | effect on the example corpus (129 questions) |
|---|---|
| Holm over leader-vs-rest | erases `recursive − semantic_no_floor` (+0.111, bootstrap p = 0.022), a result reproduced independently and guarded by `tests/test_claims.py` |
| Hansen model confidence set | retains **all five** candidates, here and on an 18-question run — less informative than the uncorrected report it replaces |

Counting wins needs no correction at all: the probabilities sum to one across the field,
so the multiplicity is priced in by construction. Calibration was measured over 1,500
simulated comparisons (3–5 candidates, correlated per-question outcomes, n ∈ {18, 30, 60,
129}): picks reported in the 90–100% band were correct 96% of the time, and in the 80–90%
band 82% — which is why the bar sits at 0.90 and not lower. The estimate converges as
evidence accumulates: for a true 0.60 vs 0.55 gap the leader's probability rises from 60%
at n = 20 to 99% at n = 1,600.

Reproduce both tables with `tests/test_selection.py` and the corpora in `examples/`.

The gate is computed on per-question **recall**, not on `balanced`: the context penalty
is an aggregate quantity with no per-question decomposition, so it cannot be bootstrapped
over questions. Read a declared tie as "these strategies retrieve equally well on the
evidence you supplied" — the ranking still orders them, and among tied strategies
`balanced` prefers the one that spends the fewest tokens, which is a defensible
tie-break even when recall cannot distinguish them.

An undecided run is common and is not a failure of the tool — but it must still carry
information. On the bundled example corpus no strategy clears the bar (the leader wins
42% of resamples), and the report says so; it also reports that three of the five remain
in play and that **both** `semantic` variants are ruled out. A field narrowed from five
to three, plus a clear rejection of the laggards, is an actionable result. "Not enough
data" on its own is not.

## Chunk-health diagnostics (retrieval-independent)

| Column | Definition |
|---|---|
| `#chunks` | Chunks produced over the corpus. |
| `med_tok` | Median chunk size in tokens (cl100k_base). |
| `%tiny` | Chunks under `eval.min_floor_tokens` (default 200) — the fragment trap: pieces too small to give an LLM usable context. |
| `%oversized` | Chunks above the strategy's `max_tokens` or the embedding model's max sequence length — such chunks are silently truncated at embedding time, so their tails are invisible to ranking. |
| `boundary` | Chunks that neither start nor end mid-sentence. |
| `tables intact` | Fraction of source tables fully contained in one chunk (`null` without tables). |
