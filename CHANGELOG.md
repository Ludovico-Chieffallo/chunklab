# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
