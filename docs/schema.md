# The `EvalReport` JSON contract

`chunklab run` writes `report.json` by serializing the `EvalReport` model verbatim
(`report.model_dump_json()`). This document is the field-by-field contract for that JSON.

**Stability policy.** The schema is versioned by the top-level `schema_version` field
(`"MAJOR.MINOR"`). Adding a field bumps MINOR; renaming, removing, or changing the meaning
of an existing field bumps MAJOR and requires a deprecation entry in the
[CHANGELOG](../CHANGELOG.md). Consumers should tolerate unknown fields.

Current version: **1.1**.

## Top level

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | str | This contract's version. |
| `corpus_summary` | object | Run provenance and inputs summary (below). |
| `strategy_results` | array | One entry per strategy, **ranked best-first** by `corpus_summary.ranking_metric` (ties: higher `mrr`, then higher `hit_rate_at_k`, then lower `chunk_health.pct_tiny`). |
| `recommendation` | str | Plain-language recommendation naming the winner and the main diagnostic reasons. |
| `warnings` | array[str] | Non-fatal issues (e.g. questions skipped for missing gold snippets). |
| `generated_at` | str | UTC ISO-8601 timestamp. The only field expected to differ between two identical runs. |
| `viz` | object \| null | Chunk-boundary visualization data for one sample document (consumed by the HTML report; same fields as `report/viz.py` models). |

## `corpus_summary`

| Field | Type | Meaning |
|---|---|---|
| `num_documents` | int | Documents loaded. |
| `documents` | array[str] | Document ids (file stems), in evaluation order (sorted by path at load time). |
| `num_questions` | int | Questions provided. |
| `num_scored_questions` | int | Questions with at least one gold snippet (only these are scored). |
| `embedding_model` | str | Embedding model name as configured. |
| `embedding_model_revision` | str \| null | Commit hash of the locally cached Hugging Face snapshot when unambiguously resolvable; `null` otherwise (e.g. fake backend, ambiguous cache). |
| `top_k` | int | Retrieval cutoff used everywhere. |
| `ranking_metric` | str | Metric used to rank `strategy_results`. |
| `seed` | int | *(1.1)* RNG seed used for bootstrap resampling. |
| `balanced_lambda` | float | *(1.1)* λ of the balanced ranking formula. |
| `queries` | object | Map question id → query text, for scored questions. |
| `chunklab_version` | str | Version of chunklab that produced the report. |
| `corpus_sha256` | str | SHA-256 over `(doc_id, text)` pairs sorted by `doc_id`, each element NUL-terminated. Order-independent fingerprint of the parsed corpus. |
| `questions_sha256` | str | SHA-256 over the canonical JSON (`model_dump_json`) of each **scored** question, sorted by id, NUL-terminated. |

## `strategy_results[]`

| Field | Type | Meaning |
|---|---|---|
| `strategy` | str | Strategy name (`fixed`, `recursive`, `semantic`, `semantic_no_floor`, `structure`). |
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

## Determinism guarantee

Two runs on identical inputs (same documents, questions, config, embedding backend and
model revision) produce identical JSON except for `generated_at`. This is enforced by
`tests/test_determinism.py`. Documents are loaded in sorted path order and retrieval uses a
stable sort, so score ties cannot reorder results between runs.
