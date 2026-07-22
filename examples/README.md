# Examples

A ready-to-run corpus so you can try chunklab in one command.

- [`sample_docs/employee_handbook.md`](sample_docs/employee_handbook.md) — a fictional, public-domain employee handbook with headings, subsections, and a benefits table.
- [`questions.example.yaml`](questions.example.yaml) — 19 questions with verbatim gold snippets (one intentionally left without gold snippets to demonstrate the skip-with-warning behavior).
- [`config.example.yaml`](config.example.yaml) — the default configuration, spelled out.

Run it:

```bash
chunklab run --docs examples/sample_docs --questions examples/questions.example.yaml
```

Then open `chunklab_report/report.html`.
