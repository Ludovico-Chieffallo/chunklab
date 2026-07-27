# Public benchmarks

## Why these exist

chunklab's own example corpus has a defect that no amount of testing fixes: the
documents, the questions and the acceptance criteria were written by the same author.
Measuring that the tool discriminates on a corpus built to make it discriminate is
circular, and it is the first thing a careful reader should distrust.

These benchmarks remove the circularity. In both datasets the questions and the gold
spans were produced by people with no connection to chunklab.

| | QASPER | CUAD v1 |
|---|---|---|
| Documents | NLP papers (full text) | Commercial contracts |
| Questions written by | NLP practitioners who had read **only the title and abstract** | Legal-review rubric, 41 fixed categories |
| Gold spans annotated by | Different annotators, who read the full paper | Lawyers (Atticus Project) |
| Reference | Dasigi et al., NAACL 2021 | Hendrycks et al., NeurIPS 2021 D&B |
| Licence | CC BY 4.0 | CC BY 4.0 |

The QASPER protocol is the valuable part: because the question writer had not read the
body, the questions do not reuse its wording. That is exactly the vocabulary mismatch a
real user creates, and the thing a self-written corpus cannot reproduce.

## Reproducing

Datasets are **not** vendored in this repository; the scripts download them into
`~/.cache/chunklab-benchmarks/` (about 30 MB total).

```bash
python scripts/benchmarks/prepare_qasper.py --out /tmp/qasper --papers 30 --seed 0
python scripts/benchmarks/run_benchmark.py  --bench /tmp/qasper --label QASPER

python scripts/benchmarks/prepare_cuad.py   --out /tmp/cuad --contracts 15 --seed 0
python scripts/benchmarks/run_benchmark.py  --bench /tmp/cuad --label CUAD
```

Sampling is seeded and applied to a sorted id list, so a given `--seed` always selects the
same papers or contracts.

**Experimental design differs by dataset, on purpose.** QASPER is evaluated as one
multi-document corpus, which is what a real RAG index looks like. CUAD asks the *same* 41
category questions of every contract, so pooling contracts would make every question
ambiguous across all of them and would measure a different task ("find a governing-law
clause in any contract"); each contract is therefore its own corpus, and per-question
recalls are pooled afterwards so a contract with 30 annotated categories weighs more than
one with 3.

## What the conversion drops, and why

Both converters report their drop rate. It is part of the result: the benchmark we run is
not identical to the benchmark as published, and the difference has to be visible.

**QASPER**, 30 papers, seed 0 — 87 questions seen, **70 kept (80.5%)**, 125 gold snippets:

| dropped | reason |
|---|---|
| 15 | unanswerable, or evidence is only a figure/table caption (not in body text) |
| 2 | evidence shorter than 5 tokens |
| 2 | no evidence span survived filtering |

**CUAD**, 15 contracts, seed 0 — 615 questions seen, **199 kept (32.4%)**, 377 spans:

| dropped | reason |
|---|---|
| 412 | category marked `is_impossible` for that contract — nothing to retrieve |
| 41 | span shorter than 5 tokens |
| 4 | no span survived filtering |

Other choices that affect the numbers: QASPER evidence is taken from **one** annotator (the
first citing body text) rather than the union across annotators, because annotators
disagree about which paragraphs support an answer and unioning would inflate `gold_total`
with disagreement rather than with chunking quality. CUAD's long "Details: …" rubric is
kept verbatim in the query, because shortening it would silently change the benchmark.

### chunklab's validator found artifacts in a published benchmark

Running `chunklab validate` over the converted QASPER set flagged two questions whose
*entire* annotated evidence is the section heading `"Experimental Setup"` — a string that
occurs in 6 of the 30 sampled papers. Any chunk containing that heading would have scored
as a hit, inflating every strategy equally. They are dropped by the 5-token floor;
`--min-gold-tokens 0` reproduces the unfiltered set.

## Results

`BAAI/bge-small-en-v1.5`, k=5, default strategies, chunklab schema 1.2.

**QASPER** — 30 papers, 70 questions:

| strategy | balanced | recall@5 | MRR | tok@5 | %tiny |
|---|---|---|---|---|---|
| `recursive` | **0.37** | 0.40 | 0.25 | 2052 | 3% |
| `structure` | 0.37 | 0.37 | 0.25 | **1279** | 42% |
| `semantic` | 0.28 | 0.34 | 0.23 | 2879 | 0% |
| `fixed` | 0.28 | 0.32 | 0.20 | 2472 | 2% |
| `semantic_no_floor` | 0.25 | 0.29 | 0.24 | 2271 | 34% |

Top-2 paired bootstrap: `recursive − structure = +0.034`, 95% CI [−0.048, +0.121] —
**indistinguishable**; ~443 questions would be needed to separate them.

**QASPER at full scale** — all 281 dev papers, 889 questions. This was run specifically to
test the estimate above, and it is the most informative result here:

| strategy | balanced | recall@5 | tok@5 | %tiny |
|---|---|---|---|---|
| `structure` | **0.175** | 0.175 | **1378** | 39% |
| `recursive` | 0.149 | 0.175 | 2107 | 3% |
| `fixed` | 0.120 | 0.159 | 2456 | 3% |
| `semantic_no_floor` | 0.124 | 0.150 | 2084 | 35% |
| `semantic` | 0.098 | 0.146 | 2718 | 0% |

`structure − recursive = +0.000`, 95% CI [−0.022, +0.023].

**The tie gate was right and the point estimate was noise.** At 70 questions `recursive`
led by +0.034 and the gate refused to call it; at 889 questions the two are *identical* to
three decimals. A tool that had reported "use `recursive`" on the smaller sample would
have shipped a recommendation built on nothing.

And the decision is still available — on the other axis. Identical recall, but `structure`
retrieves **35% fewer tokens** (1378 vs 2107). That is what the report now says:

> No winner: 'structure' and 'recursive' are statistically indistinguishable on 889 scored
> questions (recall difference +0.000, 95% CI [−0.022, +0.023] includes zero). The
> difference is too small for any realistic number of questions to separate them, so
> choose on cost instead: 'structure' retrieves 1378 tokens per question against 2107.

This run also exposed a defect: the sample-size estimator projected **1.4 billion**
questions, because the estimate grows with 1/diff² and the difference was 1.8e-5. True,
and useless to print. Estimates beyond 100,000 questions are now reported as "no realistic
number" together with the cost comparison above.

**CUAD** — 15 contracts, 199 questions:

| strategy | recall@5 | MRR | tok@5 |
|---|---|---|---|
| `structure` | **0.632** | 0.429 | 3155 |
| `semantic` | 0.603 | 0.408 | 2922 |
| `fixed` | 0.580 | 0.424 | 2393 |
| `semantic_no_floor` | 0.574 | 0.369 | 2572 |
| `recursive` | 0.554 | 0.416 | **1955** |

Top-2 paired bootstrap: `structure − semantic = +0.029`, 95% CI [−0.027, +0.084] —
**indistinguishable**; ~721 questions would be needed.

## What these results actually say

**1. There is no universal best strategy — and this is the first evidence for it that
chunklab did not author.** `recursive` ranks *first* on QASPER and *last* on CUAD. Papers
reward a strategy that packs contiguous prose; contracts reward one that respects clause
and heading boundaries. A tool that recommended one strategy globally would be wrong on
one of these two corpora.

**2. The example corpus in this repository is much easier than real data.** Recall there is
0.71–0.82; on QASPER it is 0.29–0.40. Anyone reading the README numbers should treat them
as a formatted demonstration, not as an achievable target.

**3. The statistical gate fires on independent data — and scaling up proved it right.** On
every run the top two strategies are indistinguishable at 95% confidence: 70, 199 and 889
questions. The 889-question run is the proof that this is not excessive caution: the
+0.034 lead `recursive` held at 70 questions collapsed to +0.000 at 889. This is the
central product claim — chunklab tells you *whether the difference is real* — and refusing
to name a winner was the correct call, not a missing feature.

**4. Context cost separates what recall does not.** On QASPER, `structure` reaches recall
0.37 against `recursive`'s 0.40 while retrieving **38% fewer tokens** (1279 vs 2052) — they
tie on `balanced`. On a corpus where recall is statistically tied, token cost is the
decidable difference, and it is the one that keeps costing money after the decision.

## Limitations of this exercise

- One embedding model (`bge-small-en-v1.5`) and one `k` (5). Conclusions about *strategies*
  may not transfer to other models.
- Sample sizes (30 papers, 15 contracts) were chosen for runtime, not for statistical
  power — which is precisely why both runs end in a declared tie.
- CUAD questions are legal-review rubrics repeated across contracts, not natural user
  queries; they test clause preservation, not phrasing mismatch.
- Absolute recall values are not comparable to published QASPER/CUAD numbers: those
  systems answer questions, chunklab only measures whether a chunk containing the
  annotated evidence is retrieved at k.

## Attribution

Both datasets are CC BY 4.0 and are used unmodified except for the documented conversion.

> Dasigi, Lo, Beltagy, Cohan, Smith, Gardner. *A Dataset of Information-Seeking Questions
> and Answers Anchored in Research Papers.* NAACL 2021.

> Hendrycks, Burns, Chen, Ball. *CUAD: An Expert-Annotated NLP Dataset for Legal Contract
> Review.* NeurIPS 2021 Datasets and Benchmarks.
