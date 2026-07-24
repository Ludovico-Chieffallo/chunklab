# Example corpus

Five fictional documents, each designed to stress a different axis of chunking. Every
"declared effect" below was verified by measurement during corpus construction (real
embeddings, `BAAI/bge-small-en-v1.5`, default config): a document was rewritten until its
effect showed up in the numbers, never the other way around. All content is original and
fictional (Nimbus, Vertex Analytics, Meridian, Aurora are invented organizations); no real
company material is included.

| File | Archetype | Declared, measured effect |
|---|---|---|
| `faq_support.md` | 112 short Q/A entries in 14 categories, no per-entry headings | Small chunks win; `structure` keeps ~700-token categories whole, which exceed the embedder's 512-token window, so tail entries become invisible to ranking |
| `contract_msa.md` | 15 numbered articles, multi-paragraph clauses, cross-references, key numbers far from the heading vocabulary | `structure` wins (heading context travels with the clause); `fixed` cuts mid-clause |
| `api_reference.md` | Nested H2/H3 headings, code blocks, parameter docs | `structure` wins by keeping endpoint + code together; `semantic` fragments on code blocks |
| `whitepaper.pdf` | Long flowing prose with no headings (generated PDF) | `structure` degenerates (oversized truncated blocks, worst MRR); paragraph-respecting chunks win |
| `policy_tables.docx` | HR/IT policy manual, answers inside table cells (generated DOCX) | `fixed` is the only strategy that splits tables (`table_integrity` < 1); table-aware diagnostics differentiate |

`questions.yaml` holds 129 questions with verbatim gold snippets, tagged by what they
stress (`short_answer`, `needs_context`, `boundary`, `table`, `code`, `two_sentence`,
`multi_snippet`, `cross_doc`) and by source document (`src:<name>`).

## What the corpus demonstrates

Run it yourself:

```bash
chunklab run --docs examples/corpus --questions examples/questions.yaml
```

- **No universal winner:** three different strategies win depending on the document
  (`fixed` on the FAQ, `structure` on contract/API/policy, `recursive` on the whitepaper).
  This is the point of the tool — you have to measure on *your* corpus.
- **Within single documents the choice moves recall@5 by up to ~0.28**
  (API reference: 0.84 best vs 0.56 worst); across the pooled corpus the spread compresses
  exactly *because* no strategy is safe everywhere.
- The acceptance tests for these claims live in `tests/test_corpus_discriminates.py`
  (marked `slow`; they run the real embedding model).

## Regenerating the binary documents

`whitepaper.pdf` and `policy_tables.docx` are generated, committed for convenience, and
reproducible:

```bash
python examples/generators/gen_whitepaper_pdf.py
python examples/generators/gen_policy_docx.py
```
