# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

### Fixed
- Dense retrieval now uses a stable sort, making result order deterministic when
  chunks tie on score.

## [0.1.0] - 2026-07-19

### Added
- Initial MVP: PDF/DOCX/TXT/MD loaders; `fixed`, `recursive`, `semantic` (with
  min-size floor), `semantic_no_floor`, `structure` chunkers; local
  sentence-transformers embeddings (`BAAI/bge-small-en-v1.5`); dense cosine retrieval;
  gold-snippet matching with fuzzy fallback and split-across-chunks detection;
  recall@k / hit-rate@k / MRR / precision@k; chunk-health diagnostics; console, HTML
  and JSON reports; Typer CLI (`run`, `strategies`, `demo`); `evaluate()` Python API;
  Gradio demo app.
