# Keeping chunking honest in CI

Chunking is chosen once and then quietly rots. Documents get rewritten, a loader is
upgraded, someone changes `chunk_size` to fix an unrelated problem, an export starts
emitting a byte-order mark and every heading disappears. None of that raises an error;
retrieval just gets worse, and the first sign is a user complaining about an answer.

`chunklab check` re-runs the pinned configuration and compares it against a stored report.

```bash
# once: record what "good" looks like
chunklab check --docs ./docs --questions questions.yaml \
               --baseline chunklab-baseline.json --update-baseline

# in CI: fail when it gets worse
chunklab check --docs ./docs --questions questions.yaml \
               --baseline chunklab-baseline.json
```

Exit code `1` means a regression; commit `chunklab-baseline.json` alongside your corpus.

## What makes a build fail

**The default gate is the paired bootstrap over your questions** — the same test that
gates the recommendation. A build fails when the recall drop is larger than sampling noise
can explain, i.e. when the 95% confidence interval of the difference lies entirely below
zero.

This is deliberate. A tool that refuses to name a winner on noise must not fail a build on
noise either. One question out of forty flipping is not a regression; it is a question
flipping.

Pairing requires the same questions on both sides, which is why the report records
`questions_sha256`. If you edit `questions.yaml`, the two runs are no longer paired,
chunklab says so, and the statistical gate is skipped rather than faked.

## When you want a blunt floor as well

```bash
chunklab check ... --max-drop 0.05
```

`--max-drop` fails on the raw difference regardless of significance. It is off by default,
and it is the right tool when the operational cost of a drop matters more than the
evidence for it — or when the question set changed and there is no paired test to run.

Both gates can fire; the output names whichever did.

## Tracking a specific strategy

By default `check` tracks the strategy the **baseline** ranked first — the choice the
baseline was recorded to defend. With `--compare-retrievers` in play, the retriever is
matched too, so `structure + hybrid` is compared against `structure + hybrid`.

To watch the configuration you actually deployed rather than the winner:

```bash
chunklab check ... --strategy recursive
```

## A GitHub Actions example

```yaml
name: retrieval
on: [pull_request]

jobs:
  chunking:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - run: pip install chunklab
      - uses: actions/cache@v4
        with:
          path: ~/.cache/chunklab
          key: chunklab-embeddings-${{ hashFiles('docs/**') }}
          restore-keys: chunklab-embeddings-
      - run: |
          chunklab validate --docs ./docs --questions questions.yaml
          chunklab check --docs ./docs --questions questions.yaml \
                         --baseline chunklab-baseline.json
```

Caching `~/.cache/chunklab` matters: embedding dominates the runtime, and the cache is
keyed by model revision so it can never serve vectors from different weights. Run
`chunklab validate` first — a broken gold snippet is a question-set bug, and you want it
reported as one rather than as a retrieval regression.

## When the baseline should move

Regenerate with `--update-baseline` when the drop is *intended*: you changed the corpus on
purpose, or accepted a cheaper strategy that retrieves slightly less. Review the numbers
in the diff before you do — a baseline updated to make CI green is a baseline that no
longer defends anything.
