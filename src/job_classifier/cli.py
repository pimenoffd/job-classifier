"""Command-line interface for job_classifier."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import NamedTuple

from .decision import (
    NO_MATCH,
    REVIEW_YES,
    THRESHOLD_CONFIDENT,
    THRESHOLD_MARGIN,
    THRESHOLD_MATCH,
    decide,
)
from .matcher import Candidate, ClassifierIndex, build_index, match
from .normalize import normalize

DEFAULT_RAW_POSITIONS_PATH = Path("data/raw_positions.csv")
DEFAULT_LABELED_SAMPLE_PATH = Path("data/labeled_sample.csv")

#: Column names of `labeled_sample.csv`.
COL_ID = "id"
COL_TITLE = "Исходное наименование должности"
COL_EXPECTED = "Правильный код"

#: Diagnostic sweep for the metrics table.  The shipped threshold is fixed at
#: `decision.THRESHOLD_MATCH`; these neighbours only show how sensitive the
#: metrics are to it (PLAN.md §7 step 4: "печатает таблицу метрик при разных
#: порогах").
THRESHOLD_SWEEP = (0.45, 0.50, 0.55, 0.60, 0.65)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a `;`-delimited, UTF-8-BOM CSV file into a list of dict rows."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        return list(reader)


#: Columns of the `match` command's output, in order (PLAN.md §6/§7 step 5).
RESULTS_COLUMNS = (
    "id",
    "Исходное наименование",
    "Код",
    "Наименование по классификатору",
    "Уверенность",
    "Требует проверки",
)

DEFAULT_RESULTS_PATH = Path("results.csv")


def cmd_match(args: argparse.Namespace) -> None:
    rows = read_csv_rows(args.input)
    index = build_index()

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(RESULTS_COLUMNS)
        for row in rows:
            title = row[COL_TITLE]
            normalized = normalize(title)
            candidates = match(title, index)
            decision = decide(candidates, normalized_query=normalized)
            writer.writerow(
                [
                    row[COL_ID],
                    title,
                    decision.code,
                    decision.name,
                    f"{decision.confidence:.3f}",
                    decision.requires_review,
                ]
            )


# ---------------------------------------------------------------------------
# `evaluate`: quality measurement against the 50-record labeled sample
# ---------------------------------------------------------------------------


class EvaluatedRow(NamedTuple):
    """One labeled record with its scored candidates — threshold-independent.

    Scoring happens once; `summarize()` then applies whichever `T_match` it is
    asked about, so the diagnostic sweep costs nothing extra.
    """

    row_id: str
    title: str
    expected_code: str
    normalized: str
    candidates: list[Candidate]


class Error(NamedTuple):
    row_id: str
    title: str
    predicted_code: str
    expected_code: str


class Metrics(NamedTuple):
    threshold_match: float
    total: int
    correct: int
    accuracy: float
    #: Rejection = predicting the `НЕТ СООТВЕТСТВИЯ` sentinel.  `None` when the
    #: denominator is empty (nothing predicted / nothing labeled as OOD).
    ood_precision: float | None
    ood_recall: float | None
    review_share: float
    auto_accepted: int
    auto_accepted_errors: int
    errors: list[Error]


def evaluate_rows(rows: list[dict[str, str]], index: ClassifierIndex) -> list[EvaluatedRow]:
    """Normalize and score every labeled row, without deciding anything yet."""
    evaluated = []
    for row in rows:
        title = row[COL_TITLE]
        evaluated.append(
            EvaluatedRow(
                row_id=row[COL_ID].strip(),
                title=title,
                expected_code=row[COL_EXPECTED].strip(),
                normalized=normalize(title),
                candidates=match(title, index),
            )
        )
    return evaluated


def summarize(
    evaluated: list[EvaluatedRow], threshold_match: float = THRESHOLD_MATCH
) -> Metrics:
    """Apply the decision rule at `threshold_match` and aggregate the metrics."""
    correct = 0
    review = 0
    auto_accepted = 0
    auto_accepted_errors = 0
    true_positive = 0  # predicted OOD and truly OOD
    false_positive = 0  # predicted OOD but a code was expected
    false_negative = 0  # predicted a code but truly OOD
    errors: list[Error] = []

    for row in evaluated:
        decision = decide(row.candidates, row.normalized, threshold_match=threshold_match)
        is_correct = decision.code == row.expected_code
        if is_correct:
            correct += 1
        else:
            errors.append(Error(row.row_id, row.title, decision.code, row.expected_code))

        if decision.requires_review == REVIEW_YES:
            review += 1
        else:
            auto_accepted += 1
            if not is_correct:
                auto_accepted_errors += 1

        predicted_ood = decision.code == NO_MATCH
        expected_ood = row.expected_code == NO_MATCH
        if predicted_ood and expected_ood:
            true_positive += 1
        elif predicted_ood:
            false_positive += 1
        elif expected_ood:
            false_negative += 1

    total = len(evaluated)
    predicted_ood_total = true_positive + false_positive
    expected_ood_total = true_positive + false_negative
    return Metrics(
        threshold_match=threshold_match,
        total=total,
        correct=correct,
        accuracy=correct / total if total else 0.0,
        ood_precision=true_positive / predicted_ood_total if predicted_ood_total else None,
        ood_recall=true_positive / expected_ood_total if expected_ood_total else None,
        review_share=review / total if total else 0.0,
        auto_accepted=auto_accepted,
        auto_accepted_errors=auto_accepted_errors,
        errors=errors,
    )


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:6.1f}%"


def cmd_evaluate(args: argparse.Namespace) -> None:
    rows = read_csv_rows(args.input)
    evaluated = evaluate_rows(rows, build_index())
    metrics = summarize(evaluated)

    print(f"Оценка качества на {args.input} ({metrics.total} записей)")
    print(
        f"Действующие пороги: T_match={THRESHOLD_MATCH}, "
        f"T_confident={THRESHOLD_CONFIDENT}, T_margin={THRESHOLD_MARGIN}"
    )
    print()
    print(f"Метрики при действующем пороге T_match = {THRESHOLD_MATCH}")
    print(f"  Strict accuracy         : {metrics.correct}/{metrics.total} = {_pct(metrics.accuracy)}")
    print(f"  Precision отсева        : {_pct(metrics.ood_precision)}")
    print(f"  Recall отсева           : {_pct(metrics.ood_recall)}")
    print(f"  Доля на ручную проверку : {_pct(metrics.review_share)}")
    print(
        f"  Авто-принято            : {metrics.auto_accepted} "
        f"(ошибок среди них: {metrics.auto_accepted_errors})"
    )
    print()
    print(
        "  Примечание: в размеченной выборке всего 2 истинных «НЕТ СООТВЕТСТВИЯ»,"
        " поэтому\n  precision/recall отсева статистически ненадёжны — метрика"
        " приводится, но не является\n  основанием для выводов (см. NOTE.md)."
    )
    print()

    print("Диагностическая таблица по T_match (в поставке порог фиксирован — таблица справочная)")
    print(f"  {'T_match':>8}  {'accuracy':>9}  {'precision':>10}  {'recall':>8}  {'review':>8}  {'auto-err':>8}")
    for threshold in THRESHOLD_SWEEP:
        swept = summarize(evaluated, threshold_match=threshold)
        marker = " *" if threshold == THRESHOLD_MATCH else "  "
        print(
            f"{marker}{threshold:>8.2f}  {_pct(swept.accuracy):>9}  {_pct(swept.ood_precision):>10}"
            f"  {_pct(swept.ood_recall):>8}  {_pct(swept.review_share):>8}"
            f"  {swept.auto_accepted_errors:>8}"
        )
    print("  * — действующий порог")
    print(
        "  Таблица плоская не по ошибке: после нормализации все профильные записи выборки\n"
        "  совпадают с классификатором точно (s1 = 1.000), а обе непрофильные отсекаются\n"
        "  словарём непрофильной лексики — то есть ни одна запись не лежит близко к границе."
    )
    print()

    if metrics.errors:
        print(f"Ошибки ({len(metrics.errors)}): предсказание не совпало с «{COL_EXPECTED}»")
        for error in metrics.errors:
            print(
                f"  id={error.row_id:<5} {error.title!r} -> {error.predicted_code} "
                f"(ожидалось {error.expected_code})"
            )
    else:
        print("Ошибок нет: предсказание совпало с эталоном на всех записях.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="job_classifier")
    subparsers = parser.add_subparsers(dest="command", required=True)

    match_parser = subparsers.add_parser("match", help="Match raw job titles against the classifier")
    match_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RAW_POSITIONS_PATH,
        help="Path to raw_positions.csv (default: data/raw_positions.csv)",
    )
    match_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Path to write results.csv (default: results.csv)",
    )
    match_parser.set_defaults(func=cmd_match)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Measure quality against the labeled sample"
    )
    evaluate_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_LABELED_SAMPLE_PATH,
        help="Path to labeled_sample.csv (default: data/labeled_sample.csv)",
    )
    evaluate_parser.set_defaults(func=cmd_evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
