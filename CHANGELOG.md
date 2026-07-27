# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Metric rigor (schema 1.1, additive): `retrieved_tokens_at_k` and `context_efficiency`
  per strategy expose the context cost that containment-based recall hides; new
  `balanced` ranking metric (`recall - lambda * (tokens/min_tokens - 1)`, documented and
  motivated in `docs/metrics.md`); `precision_at_k` and `tok@k` now shown in the console
  table with a one-line legend.
- Statistical honesty: per-strategy bootstrap `ci95`, and a paired-bootstrap gate on the
  recommendation — when the top-2 recall difference's 95% CI includes zero, the report
  declares the strategies statistically indistinguishable and estimates the question
  count needed to separate them, instead of naming a winner. Warning under 15 scored
  questions. Seeded via `eval.seed` (recorded in `corpus_summary`).
- Anti-degeneration guard: hidden `whole_document` strategy plus tests proving that raw
  recall rewards it (the bias) and `balanced` demotes it (the fix).

### Changed
- **`balanced` is now the default `eval.ranking_metric`** (was `recall_at_k`). On the
  example corpus this leaves the pooled ranking unchanged, but on a single-document
  corpus it reverses the top two (`structure` over `fixed`, which buys +0.03 recall at
  2.2x the context cost) — pinned by `test_balanced_changes_the_decision_on_real_data`.
  Set `ranking_metric: recall_at_k` to restore the previous behavior.
- The README quickstart now shows a declared statistical tie, because on 129 questions
  the top two strategies genuinely are within noise.

## [0.2.0] - 2026-07-24

### Added
- Discriminating example corpus (`examples/corpus/`): five fictional documents (FAQ,
  contract, API reference, heading-free PDF whitepaper, table-heavy DOCX policy manual),
  each measured to produce a distinct chunking failure mode, with 129 tagged questions
  (`examples/questions.yaml`) and generator scripts for the binary documents. Design and
  measured effects documented in `examples/CORPUS.md`.
- `tests/test_corpus_discriminates.py`: fast data-validity checks plus `slow` acceptance
  tests — aggregate recall swing >= 0.10, max per-document swing >= 0.15, and at least two
  distinct per-document winners (the "no universal best strategy" claim, now enforced).
- `EvalReport.schema_version` (`"1.0"`): the JSON report is now a versioned public
  contract, documented field-by-field in `docs/schema.md`.
- Run provenance in `corpus_summary`: `chunklab_version`, `embedding_model_revision`
  (resolved from the local Hugging Face cache when unambiguous), `corpus_sha256`,
  `questions_sha256`.
- `CHANGELOG.md`, `CONTRIBUTING.md`.
- `tests/test_determinism.py`: two runs on identical inputs produce identical JSON
  (modulo `generated_at`).
- `tests/test_readme_claims.py`: guards the README against reintroducing unsourced
  quantitative claims.

### Changed
- README: replaced the two unsourced opening statistics (retrieval-failure rate,
  accuracy swing) with a qualitative, sourced formulation (task 0.2 of the roadmap).
- `[project.urls]`: fixed `Homepage` casing; added `Repository`, `Issues`, `Changelog`.

### Changed (corpus)
- The previous single-document example (`employee_handbook.md` + 19 questions) moved to
  `tests/data/` as a pure test fixture; `examples/` now ships the discriminating corpus.

### Fixed
- Dense retrieval now uses a stable sort, making result order deterministic when
  chunks tie on score.
- `StructureChunker` no longer emits tiny heading-only chunks when a parent heading is
  immediately followed by a subheading (e.g. `## Section` directly above `### Sub`); the
  heading now merges into its first subsection. Found by measuring the API-reference
  corpus document, where 12 of 34 structure chunks were 5-token heading fragments.

## [0.1.0] - 2026-07-19

### Added
- Initial MVP: PDF/DOCX/TXT/MD loaders; `fixed`, `recursive`, `semantic` (with
  min-size floor), `semantic_no_floor`, `structure` chunkers; local
  sentence-transformers embeddings (`BAAI/bge-small-en-v1.5`); dense cosine retrieval;
  gold-snippet matching with fuzzy fallback and split-across-chunks detection;
  recall@k / hit-rate@k / MRR / precision@k; chunk-health diagnostics; console, HTML
  and JSON reports; Typer CLI (`run`, `strategies`, `demo`); `evaluate()` Python API;
  Gradio demo app.
