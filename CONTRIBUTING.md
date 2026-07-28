# Contributing to chunklab

Thanks for your interest. chunklab is deliberately narrow: it answers *which chunking
strategy retrieves best on my corpus, and why* — and nothing else. Please read
["What this is (and isn't)"](README.md#what-this-is-and-isnt) before proposing features;
PRs that turn it into a RAG framework, a parser, or an LLM-eval platform will be
declined regardless of quality.

## Development setup

```bash
git clone https://github.com/Ludovico-Chieffallo/chunklab
cd chunklab
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quality bar

- **Lint/format:** `ruff check .` and `ruff format` (line length 100). CI enforces it.
- **Tests:** `pytest` must be green. Fast tests run with the deterministic
  `FakeEmbedder` and must not download models; tests that need the real embedding
  model are marked `@pytest.mark.slow` (excluded by default, run with `pytest -m slow`).
- **One change per PR**, with the tests and docs that belong to it. No PR without tests.
- **`EvalReport` is a public contract.** Any schema change requires a
  `schema_version` bump per the policy in [docs/schema.md](docs/schema.md) and a
  CHANGELOG entry. Never rename or repurpose existing fields without a deprecation.
- **No unsourced numbers.** Quantitative claims in docs must link a source or be
  reproducible with a command in this repo (`tests/test_readme_claims.py` guards this).
- **No telemetry, no API-key requirements** in the default path. Ever.

## Reporting issues

Include: chunklab version, Python version, OS, the exact command, and — when the issue
is about scoring — the smallest document + question pair that reproduces it.

## Branch protection

`main` requires a pull request and green CI (`test (3.11)`, `test (3.12)`) before merging,
with the branch up to date. Force pushes and branch deletion are refused outright.

Administrators can bypass the pull-request rule in an emergency. That escape hatch is
deliberate and should stay rare: the reason the rule exists is that discipline alone
already failed once — PR #5 was merged two minutes before its CI finished, and passed by
luck rather than by check.
