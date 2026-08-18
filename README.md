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

## Installation

### uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### From repository

```bash
git clone https://github.com/pimenoffd/job-classifier.git
cd job-classifier
uv sync
```

### Dependencies

```bash
uv sync
```

### Data setup

The project includes sample data CSVs in `data/` directory:
- `classifier.csv` — canonical 56-class taxonomy
- `raw_positions.csv` — noisy job titles to match (300 records)
- `labeled_sample.csv` — ground truth for validation (50 records)

## Configuration

Thresholds and weights live in `config.toml` at the repo root (`[decision]`,
`[normalize]`, `[matcher]`) — edit values there to retune the pipeline, no
code changes needed. Notably `decision.out_of_scope_safeguard_enabled`
(default `false`) toggles a curated dictionary of 20 known non-construction
job titles (`data/out_of_scope_stems.txt`); off by default because it only
covers previously observed vocabulary and doesn't generalize — see `NOTE.md`
for the measured tradeoff.

Word lists the normalization pipeline depends on are plain text files under
`data/`, one `pattern;replacement` (or one stem) per line:
`abbreviations.txt`, `synonyms.txt`, `crane_technique.txt`,
`out_of_scope_stems.txt`.

## Run

```bash
uv run python -m job_classifier match      # data/raw_positions.csv -> data/results.csv (300 records)
uv run python -m job_classifier evaluate   # metrics on data/labeled_sample.csv
```

Both commands read their input from `data/` by default and accept
`--input` to override it. `match` also accepts `--output` (default
`data/results.csv`); `evaluate` has no `--output` — it prints metrics to
stdout. E.g.:

```bash
uv run python -m job_classifier match --input data/raw_positions.csv --output data/results.csv
uv run python -m job_classifier evaluate --input data/labeled_sample.csv
```

## Test

```bash
uv run pytest
```

## Output

`data/results.csv` (`;`-delimited, UTF-8): one row per input record with
columns `id`, `Исходное наименование`, `Код`, `Наименование по
классификатору`, `Уверенность`, `Требует проверки` (`да`/`нет`).
`НЕТ СООТВЕТСТВИЯ` is the literal sentinel for "no match", not a code.

`Уверенность` is always confidence **in the decision that was taken**,
so it reads the same way in both cases: low = uncertain = deserves a
look, high = settled. What the decision *is* differs — when `Код` is a
classifier code, it is confidence in that code (higher = more sure this
is the right entry); when `Код` is `НЕТ СООТВЕТСТВИЯ`, it is confidence
in the rejection (higher = more sure this really is not a construction
role). The two are calibrated onto one scale.

`Требует проверки` follows `Уверенность` only on the match side (low
confidence or a thin margin to the runner-up sends it to a human). By
default, every `НЕТ СООТВЕТСТВИЯ` requires review regardless of how high
its confidence reads — a low score is not a verified fact the way a
dictionary hit is; it can also mean the normalization vocabulary didn't
recognize an unfamiliar, genuine construction title. Set
`decision.review_all_rejections = false` in `config.toml` to only review
rejections within `decision.review_band` of the acceptance boundary and
auto-reject the rest. See `NOTE.md` §3.
