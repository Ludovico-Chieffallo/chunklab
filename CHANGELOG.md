# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **`token_spans` was quadratic on non-ASCII text.** Mapping byte offsets back to
  character offsets scanned the whole offset map whenever a token boundary fell inside a
  multi-byte character. On CJK text, where that happens at nearly every boundary, 48k
  characters took ~19 s; it now takes ~21 ms (binary search), and scales linearly.
  Output is unchanged, verified against the previous implementation on ASCII, accented,
  CJK, Cyrillic, emoji and combining-character text.
- **The abbreviation list in the sentence splitter never matched.** Punctuation was
  stripped before whitespace, so the candidate word kept its trailing dot (`dr.` was
  compared against `dr`). Every `Dr.`, `e.g.`, `Inc.` was splitting a sentence in two,
  which affected semantic chunking, boundary-health diagnostics and the bootstrapper.
- **A byte-order mark erased a markdown document's structure.** A leading U+FEFF, which
  Windows editors add invisibly, detached the first heading from the start of its line
  and left the document with *zero* elements — silently degrading `structure` chunking.
- **A non-UTF-8 document aborted the whole run** with a `UnicodeDecodeError` that did not
  say which file was at fault. Text files are now decoded UTF-8 → cp1252 → replacement,
  the encoding used is recorded in `Document.metadata`, and a fallback decode raises a
  warning in the report because mojibake degrades retrieval without erroring.
- Any loader failure is now raised as `DocumentLoadError` **naming the file**, instead of
  a bare parser traceback from the middle of a directory scan.

### Added
- **On-disk embedding cache**, on by default. Comparing five strategies re-embeds much of
  the same text, and re-running after editing questions re-embeds an unchanged corpus;
  vectors are memoized in a SQLite store under `~/.cache/chunklab/`. The key covers the
  model *and* its resolved revision, so a cache that outlives a model upgrade can never
  serve vectors from the old weights. Disable with `--no-cache`, `embedding.cache: false`
  or `CHUNKLAB_NO_CACHE=1`; relocate with `CHUNKLAB_CACHE_DIR`. Measured on the example
  corpus: 15.9 s → 7.7 s, with a byte-identical report.
- **Progress reporting during a run** (`on_progress` callback on `evaluate` /
  `run_evaluation`, wired to the CLI spinner): embedding dominates the runtime and
  silence there read as a hang.
- A corpus whose documents contain **no extractable text** is now an error rather than a
  meaningless result, and a partially empty corpus warns — the scanned-PDF case.
- **`chunklab validate --docs ... --questions ...`**: checks a question set against the
  corpus before an evaluation is spent on it. Flags gold snippets that are missing or
  only fuzzily present (printing the *verbatim* source text and its offset, ready to
  paste), duplicate ids, questions without gold, snippets present in several documents,
  and snippets too short to match meaningfully. Exits non-zero on errors so it can gate
  CI. Suggestions are suppressed below a plausibility floor and snapped to word
  boundaries, so a "fix" is never misleading.
- `docs/getting-started.md`: the real workflow, honestly timed (write questions →
  validate → run → read), with guidance on question count and hard cases.
- **`chunklab bootstrap --docs ... -n 20`**: drafts a question set from the corpus with a
  local heuristic (no API key, no LLM). Gold snippets are verbatim sentences chosen for
  stating a quantity, duration or amount; queries are template transformations, capped at
  16 words and typed (`How long`/`How much`/`How many`/`What percentage`). Every draft is
  written with `reviewed: false`, and `chunklab run` warns while unreviewed questions are
  still being scored. An optional LLM backend is planned and will stay off by default.
  The generator prefers emitting fewer questions over emitting malformed ones: it drops
  sentences whose fact sits inside a subordinate clause (the question would attach it to
  the wrong subject), predicates left dangling on a comparative, and asides between
  commas; it inverts compound auxiliaries around the subject (`can custom images be
  imported`, not `can be custom images imported`) and strands a bare `at`/`to` so the
  draft reads as English. It says so when it returns fewer than requested.
- `Question.reviewed` (default `true`, so hand-written sets are unaffected).
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
- Sentence splitting now ends a sentence when markdown emphasis follows the terminal
  punctuation (`**Is it supported?** Yes, ...`), which previously glued a FAQ heading to
  the answer that followed it — affecting semantic chunking, boundary health and the
  question bootstrapper alike.
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
