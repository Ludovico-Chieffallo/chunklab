# Getting started

The honest version of the workflow. `chunklab run` takes about a minute; building the
question set it needs takes longer, and this page is about doing that part efficiently.

```
write questions  →  chunklab validate  →  chunklab run  →  read the report
      ~30 min           seconds              ~1 min
```

## 1. Pick the documents

Point `--docs` at a directory of PDF, DOCX, TXT or MD files. Ten to fifty documents is a
good size: enough for retrieval to have to discriminate, small enough to iterate. Include
the awkward ones — the scanned-looking PDF, the spreadsheet-turned-DOCX, the wiki page
with no headings. Those are where chunking strategies actually differ.

If your corpus mixes very different document types, expect the pooled result to be less
decisive than a per-type run: a strategy that wins on contracts can lose on FAQs, and
averaging hides that. Running chunklab once per document type is often more actionable.

## 2. Write 15–30 questions with gold snippets

If a blank page is the obstacle, draft one mechanically first:

```bash
chunklab bootstrap --docs ./docs --out questions.draft.yaml -n 20
```

The heuristic backend (local, no API key) picks factual sentences — ones stating a
duration, an amount, a percentage — uses them verbatim as gold snippets, and turns each
into a draft question. **The queries are drafts**: rewrite them in your users' words,
delete the ones not worth asking, then set `reviewed: true`. `chunklab run` warns for as
long as unreviewed questions remain, because a result is only as good as the questions
behind it.


A question needs a `query` (what a user would ask) and one or more `gold_snippets` — the
passage in your document that answers it, **copied verbatim**:

```yaml
questions:
  - id: q1
    query: "How much notice do I have to give to terminate?"
    gold_snippets:
      - "written notice at least 30 days prior to termination"
    tags: [contracts]
```

Practical advice, in rough order of impact:

- **Copy, never retype.** Paste the passage from the parsed document. A paraphrase scores
  zero for every strategy, which looks like a chunking problem and is not.
- **Aim for 15–30.** Below 15, chunklab warns you: differences between strategies will not
  be statistically meaningful, and the report will decline to name a winner (see
  [metrics.md](metrics.md)).
- **Write the questions your users actually ask**, in their words — not sentences lifted
  from the document with a question mark. Retrieval that only works when the query echoes
  the source is retrieval that will fail in production.
- **Include hard cases deliberately**: answers that sit inside tables, answers that need
  the section heading to be unambiguous ("30 days" under two different sections), answers
  that straddle a paragraph boundary. These are the questions that separate strategies.
- **Use two snippets** when a full answer genuinely needs two passages; recall then
  measures partial answers correctly.

## 3. Validate before you run

```bash
chunklab validate --docs ./docs --questions questions.yaml
```

This catches the mistakes that silently ruin an evaluation — snippets that drifted from
the source, duplicate ids, snippets so short they match by accident, snippets that appear
in several documents — and for a drifted snippet it prints the **verbatim source text,
ready to paste**:

```
ERROR q1 (not_found): gold snippet not found in the corpus (closest match 81%):
'300 requests per minute on the Starter plan'
      found at api_reference:2492, verbatim source text:
      '300 requests per minute on Starter, 1,200 on'
```

It exits non-zero on errors, so it works as a CI gate on the question set itself.

## 4. Run and read the report

```bash
chunklab run --docs ./docs --questions questions.yaml
```

Read the output in this order:

1. **The recommendation.** If it declares a statistical tie, believe it: the strategies
   are indistinguishable on the evidence you supplied. Add questions, or choose on the
   secondary criteria below.
2. **`tok@k`** — the context cost. Among strategies that retrieve equally well, the one
   spending fewer tokens is cheaper and faster downstream forever.
3. **The diagnostics** — `%tiny`, `boundary`, `tables intact` — explain *why* a strategy
   lost, and often suggest the fix (raise `min_tokens`, add overlap, use structure-aware
   chunking).
4. **The HTML report** for per-question drill-down: which chunk was retrieved, at what
   rank, and whether an answer was severed across two chunks.

## 5. Iterate

Change one thing at a time — chunk size, overlap, `max_tokens` — and re-run. The JSON
report (`report.json`, a [versioned contract](schema.md)) makes it easy to diff runs
programmatically, and `corpus_summary` records the exact corpus, questions, model
revision and seed behind every number.
