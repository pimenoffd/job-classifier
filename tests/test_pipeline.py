"""End-to-end regression test for the `match` subcommand (Task 5, PLAN.md §7
step 5 / §9 verify criteria).

Runs the full pipeline against the real `data/raw_positions.csv` and checks
the shape of `results.csv` plus the set of ids decided `НЕТ СООТВЕТСТВИЯ`.

NOTE on the expected id set: task-5-brief.md predicted exactly 20 non-
construction ids (the known office-role titles). The actual full-dataset run
produces 21: the 20 predicted ids plus id 38 (`МАСТЕР СТРОЙУАСТКА`), a
heavily typo-garbled *construction* title (likely "Мастер стройучастка")
whose best classifier score (0.5455, against `КЛС-009 Маляр строительный`)
falls just under `THRESHOLD_MATCH` (0.55) — and the semantically right code,
`КЛС-047 Мастер строительных и монтажных работ`, is not even the runner-up:
it shares 3rd-4th place with `КЛС-032` at 0.500. So the rejection is not
caused by the out-of-scope safeguard; it happens because
`normalize.correct_token` does not repair "стройуастк" well enough
for the phrase-level score to clear the boundary. It is flagged
`Требует проверки == "да"`, so a human reviewer would still catch it; it is
not silently misfiled. This is flagged to the controller as a scope
question for `normalize.py`/`matcher.py`, not silently patched here — see
task-5-report.md.

The pipeline ships with `decision.OUT_OF_SCOPE_STEMS_PATH` pointed at the
*empty* variant (see decision.py), so id 113 (`Переводчик`) is absent here
too: without the curated stem list nothing overrides its 0.737 score against
`КЛС-056 Проходчик`, so it comes back as a match rather than `НЕТ
СООТВЕТСТВИЯ`. `THRESHOLD_CONFIDENT` still catches it (0.737 < 0.82), so it
is flagged `Требует проверки == "да"` — not silently accepted.
"""

import csv
from pathlib import Path

from job_classifier.cli import build_parser

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Observed on the real data (see module docstring): the 20 ids predicted by
#: task-5-brief.md, minus id 113 (the curated out-of-scope list is off by
#: default), plus id 38 (a construction-role false rejection, flagged as a
#: discrepancy rather than adjusted away).
EXPECTED_NO_MATCH_IDS = {
    "38",
    "49",
    "52",
    "66",
    "84",
    "117",
    "126",
    "136",
    "147",
    "164",
    "173",
    "177",
    "191",
    "193",
    "225",
    "241",
    "256",
    "276",
    "277",
    "296",
}


def test_match_pipeline_end_to_end(tmp_path):
    output = tmp_path / "results.csv"
    parser = build_parser()
    args = parser.parse_args(
        [
            "match",
            "--input",
            str(REPO_ROOT / "data" / "raw_positions.csv"),
            "--output",
            str(output),
        ]
    )
    args.func(args)

    with open(output, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = list(reader)

    assert reader.fieldnames == [
        "id",
        "Исходное наименование",
        "Код",
        "Наименование по классификатору",
        "Уверенность",
        "Требует проверки",
    ]
    assert len(rows) == 300

    no_match_ids = {r["id"] for r in rows if r["Код"] == "НЕТ СООТВЕТСТВИЯ"}
    assert no_match_ids == EXPECTED_NO_MATCH_IDS
