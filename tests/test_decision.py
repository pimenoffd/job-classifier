"""Tests for the decision layer (docs/PLAN.md §7 step 4, §4.3, §9).

Three groups:

1. The pure threshold logic of `make_decision` (the fixed pseudocode from
   task-4-brief.md, with §4.3's calibrated OOD confidence formula).
2. The out-of-scope safeguard (controller ruling 1): the 20 known
   non-construction titles must return `НЕТ СООТВЕТСТВИЯ` regardless of
   score, and the safeguard must not fire on any of the 56 classifier
   names.
3. The end-to-end regression gate (controller ruling 2 / PLAN.md §9):
   strict accuracy >= 0.95 on `data/labeled_sample.csv` **and** zero
   errors among auto-accepted records (`requires_review == "нет"`).
"""

from pathlib import Path

import pytest

from job_classifier.cli import (
    DEFAULT_LABELED_SAMPLE_PATH,
    evaluate_rows,
    read_csv_rows,
    summarize,
)
from job_classifier.decision import (
    NO_MATCH,
    REVIEW_NO,
    REVIEW_YES,
    S_FLOOR,
    THRESHOLD_MATCH,
    is_out_of_scope,
    make_decision,
)
from job_classifier.dictionaries import load_classifier
from job_classifier.matcher import build_index
from job_classifier.normalize import normalize

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

#: The 20 non-construction titles of PLAN.md §2, verbatim as they appear in
#: `data/raw_positions.csv` — typos (`Агроом`, `Диспетче`, `Кладощвик`)
#: included, because the safeguard has to survive them.
OFFICE_TITLES = [
    "Агроом",
    "Бухгалтер материального стола",
    "Специалист отдела кадров",
    "Охранник",
    "Переводчик",
    "Повар столовой",
    "Делопроизводитель",
    "Маркетолог",
    "Диспетче автотранспорта",
    "Системный администратор 2 категории",
    "Казначей",
    "КУРЬЕР 4 РАЗРЯДА",
    "Менеджер по продажам",
    "  Программист 1С 2 категории ",
    "Экономист планового отдела",
    "Фельдшер здравпункта",
    "Кладощвик центрального склада",
    "психолог",
    "Юрисконсульт",
    "Уборщик служебных помещений",
]


@pytest.fixture(scope="module")
def index():
    return build_index()


@pytest.fixture(scope="module")
def evaluated(index):
    rows = read_csv_rows(DATA_DIR / "labeled_sample.csv")
    return evaluate_rows(rows, index)


# ---------------------------------------------------------------------------
# 1. Threshold logic
# ---------------------------------------------------------------------------


def test_ood_below_threshold_returns_sentinel_and_empty_name():
    decision = make_decision("КЛС-005", "Каменщик", 0.375, 0.300)
    assert decision.code == NO_MATCH
    assert decision.name == ""


def test_ood_confidence_is_calibrated_against_the_boundary():
    # PLAN.md §4.3's worked example: `Казначей` at 0.375 -> ~0.88, not 0.625.
    expected = round((THRESHOLD_MATCH - 0.375) / (THRESHOLD_MATCH - S_FLOOR), 3)
    assert make_decision("КЛС-005", "Каменщик", 0.375, 0.300).confidence == expected
    assert expected == pytest.approx(0.875)


def test_ood_confidence_near_the_boundary_tends_to_zero():
    decision = make_decision("КЛС-046", "Прораб", 0.549, 0.400)
    assert decision.confidence == pytest.approx(0.005, abs=1e-3)
    # ...and a near-boundary rejection still goes to a human.
    assert decision.requires_review == REVIEW_YES


def test_ood_confidence_is_clamped_to_one():
    assert make_decision("КЛС-046", "Прораб", 0.10, 0.05).confidence == 1.0


def test_ood_far_below_boundary_is_auto_accepted():
    assert make_decision("КЛС-046", "Прораб", 0.30, 0.20).requires_review == REVIEW_NO


def test_confident_match_is_auto_accepted():
    decision = make_decision("КЛС-004", "Бетонщик", 0.95, 0.60)
    assert decision == ("КЛС-004", "Бетонщик", 0.95, REVIEW_NO)


def test_match_below_confident_threshold_requires_review():
    assert make_decision("КЛС-004", "Бетонщик", 0.78, 0.40).requires_review == REVIEW_YES


def test_match_with_thin_margin_requires_review():
    # 0.90 clears T_confident, but the runner-up is 0.05 behind (< T_margin).
    assert make_decision("КЛС-004", "Бетонщик", 0.90, 0.85).requires_review == REVIEW_YES


# ---------------------------------------------------------------------------
# 2. Out-of-scope safeguard (controller ruling 1)
# ---------------------------------------------------------------------------


def test_perevodchik_is_rejected_despite_scoring_above_threshold():
    # Known residual leak: `Переводчик` scores 0.737 against КЛС-056 Проходчик
    # (a character-level coincidence). The safeguard must override the score.
    decision = make_decision(
        "КЛС-056", "Проходчик", 0.737, 0.500, normalized_query=normalize("Переводчик")
    )
    assert decision.code == NO_MATCH


def test_all_twenty_known_office_titles_are_rejected():
    rejected = [t for t in OFFICE_TITLES if is_out_of_scope(normalize(t))]
    assert rejected == OFFICE_TITLES


def test_safeguard_does_not_fire_on_any_classifier_name():
    fired = [name for _code, name in load_classifier() if is_out_of_scope(normalize(name))]
    assert fired == []


def test_safeguard_does_not_fire_on_genuine_construction_rows():
    rows = read_csv_rows(DATA_DIR / "raw_positions.csv")
    titles = [row["Исходное наименование должности"] for row in rows]
    rejected = [t for t in titles if is_out_of_scope(normalize(t))]
    # Exactly the 20 non-construction rows of PLAN.md §2, none of the 280 others.
    assert len(rejected) == 20


# ---------------------------------------------------------------------------
# 3. Regression gate (controller ruling 2 / PLAN.md §9)
# ---------------------------------------------------------------------------


def test_default_labeled_sample_path_exists():
    assert (DATA_DIR / DEFAULT_LABELED_SAMPLE_PATH.name).exists()


def test_labeled_sample_accuracy_at_least_095(evaluated):
    metrics = summarize(evaluated)
    assert metrics.total == 50
    assert metrics.accuracy >= 0.95


def test_zero_errors_among_auto_accepted_records(evaluated):
    metrics = summarize(evaluated)
    assert metrics.auto_accepted > 0
    assert metrics.auto_accepted_errors == 0
