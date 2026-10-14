# The `EvalReport` JSON contract

`chunklab run` writes `report.json` by serializing the `EvalReport` model verbatim
(`report.model_dump_json()`). This document is the field-by-field contract for that JSON.

**Stability policy.** The schema is versioned by the top-level `schema_version` field
(`"MAJOR.MINOR"`). Adding a field bumps MINOR; renaming, removing, or changing the meaning
of an existing field bumps MAJOR and requires a deprecation entry in the
[CHANGELOG](https://github.com/Ludovico-Chieffallo/chunklab/blob/main/CHANGELOG.md). Consumers should tolerate unknown fields.

Current version: **1.5**.

## Top level

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | str | This contract's version. |
| `corpus_summary` | object | Run provenance and inputs summary (below). |
| `strategy_results` | array | One entry per strategy, **ranked best-first** by `corpus_summary.ranking_metric` (ties: higher `mrr`, then higher `hit_rate_at_k`, then lower `chunk_health.pct_tiny`). |
| `recommendation` | str | The verdict in English. It names a winner only when one clears the selection bar, which most runs do not; otherwise it says what is still in play, what is ruled out and what would settle it. A program should read `verdict` instead. |
| `verdict` | object \| null | *(1.5)* The same decision as data, below. `null` in reports written before 1.5. |
| `warnings` | array[str] | Non-fatal issues (e.g. questions skipped for missing gold snippets). |
| `generated_at` | str | UTC ISO-8601 timestamp. The only field expected to differ between two identical runs. |
| `viz` | object \| null | Chunk-boundary visualization data for one sample document (consumed by the HTML report; same fields as `report/viz.py` models). |

## `corpus_summary`

| Field | Type | Meaning |
|---|---|---|
| `num_documents` | int | Documents loaded. |
| `num_documents_with_text` | int | *(1.4)* Documents that yielded extractable text and therefore contributed chunks. Lower than `num_documents` when a scanned PDF, or any file the loader could not read text from, was included: those load and count but contribute nothing to the index. |
| `documents` | array[str] | Document ids, in evaluation order (sorted by path at load time). An id is the file name without its extension; files whose names collide take their path under the scanned folder instead (`2023/msa.md`, `2024/msa.md`). |
| `num_questions` | int | Questions provided. |
| `num_scored_questions` | int | Questions with at least one gold snippet (only these are scored). |
| `embedding_model` | str | Embedding model name as configured. |
| `embedding_model_revision` | str \| null | Commit hash of the locally cached Hugging Face snapshot when unambiguously resolvable; `null` otherwise (e.g. fake backend, ambiguous cache). |
| `top_k` | int | Retrieval cutoff used everywhere. |
| `fuzzy_threshold` | float | *(1.5)* Similarity above which a gold snippet counts as found by fuzzy matching (`eval.fuzzy_threshold`). `chunklab check` refuses to compare two runs scored under different thresholds. |
| `ranking_metric` | str | Metric used to rank `strategy_results`. |
| `retrieval_modes` | array[str] | *(1.3)* Retrievers evaluated (`dense`, `bm25`, `hybrid`). More than one means `strategy_results` is a strategy x retriever matrix. |
| `seed` | int | *(1.1)* RNG seed used for bootstrap resampling. |
| `balanced_lambda` | float | *(1.1)* λ of the balanced ranking formula. |
| `embedding_prefixes` | bool | *(1.2)* Whether the model's trained query/passage instruction prefixes were applied. |
| `detected_languages` | object | *(1.2)* Map document id → detected language (ISO code) or non-Latin script name. Documents whose language could not be determined confidently are omitted, so this map may be empty or partial. Advisory only: it drives a warning, never a decision. |
| `queries` | object | Map question id → query text, for scored questions. |
| `chunklab_version` | str | Version of chunklab that produced the report. |
| `corpus_sha256` | str | SHA-256 over `(doc_id, text)` pairs sorted by `doc_id`, each element NUL-terminated. Order-independent fingerprint of the parsed corpus. |
| `questions_sha256` | str | SHA-256 over the canonical JSON (`model_dump_json`) of each **scored** question, sorted by id, NUL-terminated. |

## `strategy_results[]`

| Field | Type | Meaning |
|---|---|---|
| `strategy` | str | Strategy name (`fixed`, `recursive`, `semantic`, `semantic_no_floor`, `structure`). |
| `retriever` | str | *(1.3)* Retriever that produced these numbers (`dense`, `bm25`, `hybrid`). Defaults to `dense`; with a matrix, `(strategy, retriever)` is the unique key of an entry. |
| `config` | object | Parameters the strategy ran with. |
| `recall_at_k` | float | Mean over questions of gold snippets found in top-k / total gold snippets. |
| `hit_rate_at_k` | float | Fraction of questions with ≥ 1 gold snippet found in top-k. |
| `mrr` | float | Mean reciprocal rank of the first hit (0 when no hit in top-k). |
| `precision_at_k` | float | Mean fraction of the top-k retrieved chunks that contain a gold snippet. |
| `retrieved_tokens_at_k` | float | *(1.1)* Mean total tokens retrieved per question — downstream context cost. |
| `context_efficiency` | float | *(1.1)* Mean found-gold tokens / retrieved tokens. |
| `balanced_score` | float | *(1.1)* `recall_at_k` penalized by relative context cost; formula in docs/metrics.md. |
| `ci95` | [float, float] \| null | *(1.1)* Bootstrap 95% CI of mean per-question recall. |
| `chunk_health` | object | Retrieval-independent diagnostics (below). |
| `per_question` | array | Per-question outcomes (below). |

### `chunk_health`

| Field | Type | Meaning |
|---|---|---|
| `num_chunks` | int | Chunks produced over the whole corpus. |
| `tokens_min` / `tokens_median` / `tokens_mean` / `tokens_max` | number | Token-count stats (cl100k_base tokenizer). |
| `pct_tiny` | float | Fraction of chunks under `eval.min_floor_tokens` (default 200) — the "fragment trap" signal. |
| `pct_oversized` | float | Fraction of chunks above the strategy's `max_tokens` or the embedding model's max sequence length (truncation risk). |
| `boundary_health` | float | Fraction of chunks that neither start nor end mid-sentence. |
| `table_integrity` | float \| null | Fraction of source tables fully contained in a single chunk; `null` when the corpus has no detected tables. |
| `token_histogram` | array[[int, int]] | `(bucket_start, count)` pairs, bucket width 100 tokens. |

### `per_question[]`

| Field | Type | Meaning |
|---|---|---|
| `question_id` | str | Question id. |
| `strategy` | str | Strategy name (redundant, for flat consumption). |
| `retrieved` | array | Top-k retrieved chunks with `score`, `rank`, `is_hit`, and the full `chunk` (id, doc_id, text, token_count, char_span, strategy, contains_table, section_path). |
| `hit` | bool | ≥ 1 gold snippet found in top-k. |
| `first_hit_rank` | int \| null | Rank of the first hit. |
| `split_across_chunks` | bool | No single chunk contained a missing gold snippet, but joining two adjacent retrieved chunks would — the chunker severed the answer. |
| `gold_found_count` / `gold_total` | int | Gold snippets found / total for this question. |
| `found_gold_indices` | array[int] | *(1.1)* Indices (into the question's `gold_snippets`) of the snippets found. |

## `verdict` *(1.5)*

`chunking` always, and `retrieval` when retrievers were compared (`null` otherwise). Each
is an axis verdict:

| Field | Type | Meaning |
|---|---|---|
| `axis` | str | `chunking` or `retrieval`. |
| `decided` | bool | The leader came out best in at least 90% of paired bootstrap resamples (`SELECTION_CONFIDENCE`), and is then `winner`. |
| `winner` | str \| null | The strategy or retriever to use; `null` when undecided. |
| `leader` | str | Ranked first, whether or not it is the winner. |
| `probability_best` | object | Name → share of paired bootstrap resamples it came out best in. Sums to 1. |
| `in_play` | array[str] | Candidates the evidence does not rule out, best-first. A candidate is ruled out only on evidence against it, so a decided axis can list more than its winner. |
| `ruled_out` | array[str] | Outside the fewest candidates whose `probability_best` adds up to 95%, **and** at least as good as the leader in under 2.5% of paired resamples. |
| `margin` | object \| null | `runner_up`, `difference` (leader minus runner-up, mean per-question recall) and its paired `ci95`. |
| `questions_to_separate` | int \| null | Rough number of scored questions that would separate the top two; `null` when decided, or when no realistic number would. |
| `held` | str \| null | With a matrix, the value of the other axis this one was measured under: the best performer there, itself chosen from these data. `null` with one retriever. |

A CI step that only needs to know whether there is a winner:

```bash
jq -e '.verdict.chunking.decided' chunklab_report/report.json
```

## Determinism guarantee

Two runs on identical inputs (same documents, questions, config, embedding backend and
model revision) produce identical JSON except for `generated_at`. This is enforced by
`tests/test_determinism.py`. Documents are loaded in sorted path order and retrieval uses a
stable sort, so score ties cannot reorder results between runs.
