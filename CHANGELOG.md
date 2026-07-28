# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-07-28

Everything below shipped after 0.2.0. The headline is not a feature: several of these
entries are silent failures found by running the tool on real documents and on corpora
this project did not author.

### Fixed
- **A stale claim in the README was false.** It said both `semantic` variants were
  statistically separated from the top three strategies (CIs "+0.092 and +0.115"), and
  quoted token costs of 2,075 and 2,431. Re-measured: only `semantic_no_floor` is
  separated from all three (+0.111, +0.099, +0.103); `semantic` is not distinguishable
  from `recursive` or `structure`. The token figures were pre-prefix values. Corrected,
  and now guarded.

### Added
- **Every quantitative claim on the front page must have a source** (`tests/test_claims.py`,
  replacing `test_readme_claims.py`). This project has shipped a false number twice, both
  times written from memory and caught by hand. Each decimal, percentage or
  thousands-separated magnitude in the README and docs index is listed with the command
  that reproduces it; adding one without a source fails, and so does keeping an entry for
  a claim no longer made. Inline code is deliberately **not** exempt — a number in
  backticks is still a claim. Pages carrying measured tables must contain a Reproducing
  section with a runnable command.
- **`ruff format` is applied and gated in CI**, in an isolated commit (12 files, no
  behaviour change).
- **`chunklab check`** — a CI gate for retrieval regressions. Re-runs the pinned
  configuration and compares it against a stored `report.json`; exit 1 on regression.
  The default gate is the **paired bootstrap over questions**, the same test that gates
  the recommendation: a build fails when the drop is larger than sampling noise explains,
  not when a number moved. A tool that refuses to name a winner on noise must not fail a
  build on noise either. `--max-drop` adds an opt-in blunt floor for teams that want one,
  and is also what applies when the question set changed and no paired test is valid
  (detected via `questions_sha256`, not assumed). `--update-baseline` records a new
  baseline. Guide in `docs/ci.md`.
- **Plugin API for custom chunking strategies** (`chunklab.plugins.register_chunker`, plus
  automatic discovery of the `chunklab.chunkers` entry point). chunklab can only rank the
  strategies it can see, and a comparison that excludes the splitter your pipeline
  actually uses is beside the point. A chunker is anything with `chunk(document)`; the
  factory receives `params` as keywords and `embedder=` only if its signature asks for
  one. Built-in names cannot be shadowed and re-registration needs `replace=True`,
  because a silent override would change what a config file means with no sign of it in
  the report. Guide in `docs/extending.md`.
- **`chunklab cache`** — shows where cached embeddings live, how many vectors and how much
  disk they use; `--clear` removes them. The cache had no way to inspect or bound it.
- **Documentation site**: <https://ludovico-chieffallo.github.io/chunklab/>
  (`mkdocs.yml`, `docs/index.md`, mkdocs-material). The build runs in `--strict` mode on
  any PR touching docs, so broken links fail review; deployment happens on push to `main`,
  never from a pull request, so a fork cannot publish to the live site. `mkdocs<2` is
  pinned: the Material team warns that MkDocs 2.0 is currently unlicensed and removes the
  plugin system.

### Added (earlier in this cycle)
- **BM25 and hybrid retrieval, and a strategy × retriever matrix.** `retrieval.mode`
  already accepted `bm25` and `hybrid` in the config schema while only `dense` existed —
  a promise the configuration file made and the code did not keep. Both are now real.
  - **`bm25`**: Okapi BM25 implemented in-tree rather than added as a dependency, because
    owning it means owning the tokenizer — which is what decides whether the retriever
    works outside English. Word tokens, falling back to **character bigrams** for scripts
    written without spaces (Han, Kana, Hangul, Thai), where a word tokenizer emits one
    token per sentence and matches nothing.
  - **`hybrid`**: Reciprocal Rank Fusion (`Σ 1/(60 + rank)`, Cormack et al. 2009). Fuses
    ranks rather than scores, since unbounded idf sums and cosines cannot be summed
    without per-corpus calibration chunklab has no honest way to perform.
  - **`--compare-retrievers`** / `retrieval.compare` evaluates every strategy under every
    retriever. `balanced` then normalizes on the cheapest cell of the whole matrix, so
    the context penalty prices retrievers against each other too. `hybrid` reuses the
    dense and BM25 indexes it fuses, so asking for all three costs no more than two.
  - Measured on the example corpus, switching `dense` → `hybrid` improved recall for
    **all five** strategies (+0.098 to +0.163, every CI excluding zero) — more than the
    spread between strategies. On QASPER the same comparison is significant for **two of
    five**. The example corpus almost certainly overstates the gain: its questions and its
    documents share an author, so they share vocabulary, which is exactly what lexical
    matching rewards. Documented in `docs/metrics.md` with both tables.
- **Schema 1.3** (additive): `strategy_results[].retriever` and
  `corpus_summary.retrieval_modes`.

### Fixed
- **Hybrid fusion was order-dependent on ties.** Two retrievers that swap ranks 1 and 3
  give both chunks an identical fused score; breaking that tie by insertion order made the
  ranking depend on the order the retrievers happened to be listed in, so the same corpus
  could rank differently for no reason. Ties are now broken by best rank achieved and then
  chunk id, both intrinsic to the chunk. Caught by a test written before the measurement.
- **`structure` chunking silently degenerated on real documents.** Found by running the
  loaders over a real 10-K filing and a real annual report; in both cases it kept
  producing chunks, they were simply packed to `max_tokens` with no regard for the
  document's sections, and nothing said so.
  - **DOCX files carrying no heading styles.** The annual report had 961 paragraphs, every
    one styled `Normal (Web)` — a shape typical of anything converted from HTML — while
    being visually structured with 171 bold headings. Short, fully-bold paragraphs that do
    not end in sentence punctuation are now recognised as headings (all-caps reads as one
    level higher), and the count is recorded in `Document.metadata["inferred_headings"]`.
    On that document `structure` went from 98 chunks of ~742 tokens to 200 chunks
    following the real sections.
  - **Outlines shifted deeper by a converter.** The split depth was hard-coded at H1–H3,
    but pymupdf4llm assigned level 5 to 261 of the 268 headings in the 10-K, so three
    headings were recognised in a 66k-token filing. The depth now follows the document's
    most common heading level (never shallower than H3), which leaves conventional
    outlines — including the whole example corpus — completely unchanged.
  - A corpus whose documents have **no detectable headings at all** now warns that
    `structure`'s score reflects that fallback rather than structure-aware chunking.
- `chunklab bootstrap` emitted several questions with an identical query when a document
  repeats boilerplate ("The information required by this Item will be included…" appears
  under several headings in a 10-K). Identical queries retrieve identical chunks and add
  no evidence; only the highest-scoring one is kept.

### Added
- **Public benchmarks** (`docs/benchmarks.md`, `scripts/benchmarks/`): QASPER and CUAD v1,
  both CC BY 4.0, downloaded on demand rather than vendored. They exist to remove a defect
  no amount of testing fixes — the example corpus, its questions and its acceptance
  criteria were written by the same author. In both datasets the questions and gold spans
  come from people unconnected to this project. Every conversion choice and drop rate is
  reported, because the benchmark we run is not identical to the benchmark as published.
  - `chunklab validate` found artifacts in a *published* benchmark: two QASPER annotations
    whose entire evidence is the section heading "Experimental Setup", a string occurring
    in 6 of 30 sampled papers.
  - Headline results: `recursive` ranks **first** on QASPER and **last** on CUAD, the first
    independent evidence for the no-universal-winner claim. Recall on the bundled example
    corpus (0.71–0.82) is far above QASPER (0.15–0.40): the example corpus is a formatted
    demonstration, not an achievable target.

### Fixed
- **The sample-size estimator printed nonsense on near-ties.** The projection grows with
  1/diff², so a difference of 1.8e-5 across 889 questions was reported as "roughly
  1,449,586,924 scored questions would be needed" — true, and useless. Estimates beyond
  100,000 questions are now reported as "no realistic number would separate them", and the
  recommendation names the cheaper strategy by context cost instead, which is the decision
  actually available.

### Changed
- **README repositioned around the claim the evidence supports.** chunklab's product is
  that it tells you *whether the difference between strategies is real* and refuses to
  name a winner when it is not. The 889-question QASPER run is the justification: at 70
  questions `recursive` led `structure` by +0.034 and the gate declined to call it; at 889
  the difference is +0.000. The removed opening claim ("three different strategies win
  depending on the document") had also become false — the example corpus now has two
  distinct per-document winners, not three.
- **Schema 1.2** (additive): `corpus_summary.detected_languages` and
  `corpus_summary.embedding_prefixes`.
- **Instruction prefixes are now applied by default.** Queries and passages are embedded
  through separate entry points (`embed` / `embed_queries`), and each model family gets
  the scheme it was trained with. This changes results: on the example corpus every
  strategy's recall moved by 0.000 to +0.047, and the ranking changed. Measured honestly
  with the tool's own paired bootstrap, the improvement is significant for `recursive`
  only (+0.047, 95% CI [+0.008, +0.093]); for the other four strategies the point estimate
  is positive but statistically indistinguishable from zero on 129 questions. It is on by
  default because it is what the models were trained for and it never measured worse —
  not because the corpus-wide gain is proven. `embedding.prefixes: false` restores the
  old behaviour.
- The slow test asserting that `balanced` *overturns the winner* on the API-reference
  document no longer held once prefixes were applied, and was re-anchored to what is
  still true and still meaningful: `balanced` ranks `structure` above `fixed` despite
  `fixed`'s higher recall, because it retrieves 15% less context. On this corpus no
  recall gap is now small enough for the penalty to flip first place — which is the
  documented intent of λ = 0.05, not a regression.

### Added
- **Multilingual support.**
  - Sentence segmentation now recognises non-ASCII terminators (`。`, `｡`, `！`, `？`,
    `।`, `॥`, `؟`, `۔`, Armenian and Ethiopic stops). Scripts that do not put a space
    after the terminator previously collapsed an entire document into a **single
    sentence**, making semantic and structure chunking meaningless on Chinese, Japanese
    and Hindi. A CJK terminator inside a closing bracket is treated as quoted speech.
  - Query/passage instruction prefixes per model family: E5 (`query: ` / `passage: `,
    mandatory — omitting them degrades retrieval with no error), BGE English and Chinese
    (query-side retrieval instruction), none for unrecognised models.
  - Corpus/model mismatch warning: an English-only model over a non-English corpus is
    reported explicitly, with `intfloat/multilingual-e5-small` suggested. Detection is
    dependency-free (Unicode-range script counting plus stop-word frequency for seven
    Latin-script languages) and conservative — when the evidence is thin it claims
    nothing, because a wrong warning is worse than no warning.
- The embedding cache key now covers the query/passage role and the prefix scheme, so
  turning prefixes off or embedding the same text on the other side can never reuse the
  wrong vector.

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
