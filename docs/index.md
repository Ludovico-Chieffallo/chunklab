# chunklab

**Tells you whether the difference between chunking strategies is real — on your own
documents — before you commit to one.**

Chunking differences are usually small and noisy, and a few dozen questions cannot tell a
real 3-point gap from a coin flip. Most comparisons declare a winner anyway. chunklab's
job is to stop you from doing that.

```bash
pip install chunklab
chunklab run --docs ./docs --questions questions.yaml
```

## Where to go

| | |
|---|---|
| [Getting started](getting-started.md) | The real workflow, honestly timed: write questions → validate → run → read. |
| [Metrics and honesty](metrics.md) | What `balanced` is, why it is the default, how the statistical gate works, and what the retriever comparison measured. |
| [Public benchmarks](benchmarks.md) | Results on QASPER and CUAD — corpora this project did not author — with every conversion choice and drop rate reported. |
| [Use in CI](ci.md) | `chunklab check`: fail a build when retrieval regresses, without failing it on noise. |
| [Add your own strategy](extending.md) | The plugin interface, and how to evaluate a LangChain splitter with it. |
| [JSON schema](schema.md) | The versioned `report.json` contract, field by field. |

## What it will not do

Non-goals, kept deliberately:

- It is **not a RAG framework**, a vector database, or a document parser.
- It does **not** evaluate LLM answer quality — only whether the answer-bearing text is
  retrieved.
- It sends **no telemetry**, in any form, and needs no API key on the default path.
- It is not built for thousands of documents.

## The claim, and its evidence

On [QASPER](benchmarks.md) — 889 questions written by people who had read only each
paper's abstract, with evidence annotated separately — the top two strategies differ by
`+0.000` recall. At 70 questions one of them led by `+0.034`: noise that a less careful
tool would have shipped as a recommendation.

They are not equivalent, though. One retrieves **35% fewer tokens** for the same recall,
on every query, forever — and that is the decision worth making.
