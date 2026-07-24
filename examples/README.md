# Examples

A ready-to-run corpus so you can try chunklab in one command:

```bash
chunklab run --docs examples/corpus --questions examples/questions.yaml
```

Then open `chunklab_report/report.html`.

- [`corpus/`](corpus/) — five fictional documents (MD, PDF, DOCX), each built to stress a
  different chunking failure mode. See [`CORPUS.md`](CORPUS.md) for the design and the
  measured effects.
- [`questions.yaml`](questions.yaml) — 129 questions with verbatim gold snippets, tagged by
  what they stress and by source document.
- [`generators/`](generators/) — scripts that reproduce the binary documents
  (`whitepaper.pdf`, `policy_tables.docx`).
- [`config.example.yaml`](config.example.yaml) — the default configuration, spelled out.
