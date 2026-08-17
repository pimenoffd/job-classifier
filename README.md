# job-classifier

Matches noisy Russian job-title strings from an external 1C export
against a 56-entry canonical classifier, returning code + canonical
name + confidence per record, and flags low-confidence matches for
manual review.

Rule-based normalization + lexical fuzzy scoring — no embeddings, no
LLMs, no external APIs, CPU-only. See `NOTE.md` for the approach and
its known limitations.

## Requirements

- Python >= 3.11
- [`uv`](https://docs.astral.sh/uv/) for dependency management

## Install

```bash
uv sync
```

`data/` is gitignored, so before running either command — or the test
suite, every file of which reads from it — populate it with the three
CSVs supplied with the assignment: `classifier.csv`,
`raw_positions.csv` and `labeled_sample.csv`.

## Run

```bash
uv run python -m job_classifier match      # data/raw_positions.csv -> results.csv (300 records)
uv run python -m job_classifier evaluate   # metrics on data/labeled_sample.csv
```

Both commands read their input from `data/` by default and accept
`--input` to override it. `match` also accepts `--output` (default
`results.csv`); `evaluate` has no `--output` — it prints metrics to
stdout. E.g.:

```bash
uv run python -m job_classifier match --input data/raw_positions.csv --output results.csv
uv run python -m job_classifier evaluate --input data/labeled_sample.csv
```

## Test

```bash
uv run pytest
```

## Output

`results.csv` (`;`-delimited, UTF-8): one row per input record with
columns `id`, `Исходное наименование`, `Код`, `Наименование по
классификатору`, `Уверенность`, `Требует проверки` (`да`/`нет`).
`НЕТ СООТВЕТСТВИЯ` is the literal sentinel for "no match", not a code.

`Уверенность` is always confidence **in the decision that was taken**,
so it reads the same way in both cases: low = uncertain = deserves a
look, high = settled. What the decision *is* differs — when `Код` is a
classifier code, it is confidence in that code (higher = more sure this
is the right entry); when `Код` is `НЕТ СООТВЕТСТВИЯ`, it is confidence
in the rejection (higher = more sure this really is not a construction
role). The two are calibrated onto one scale, so `Требует проверки`
follows from a low value in either case.
