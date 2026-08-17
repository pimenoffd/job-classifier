"""Tests for the fuzzy matcher (docs/PLAN.md §7 step 3).

Score formula per the controller ruling in task-3-brief.md (overrides the
plan's literal `WRatio` wording, which restates the original architecture
doc rather than the formula §3's calibration experiment actually used):

    score = 0.5 * rapidfuzz.fuzz.token_sort_ratio(q, c) / 100
          + 0.5 * rapidfuzz.fuzz.ratio(q, c) / 100

All five mandatory cases from the §7 step 3 verify line are covered below.
"""

import pytest

from job_classifier.dictionaries import load_classifier
from job_classifier.matcher import build_index, match


@pytest.fixture(scope="module")
def index():
    return build_index()


def test_exact_match_scores_one(index):
    for code, name in load_classifier():
        candidates = match(name, index, k=1)
        assert candidates[0].code == code
        assert candidates[0].score == pytest.approx(1.0)


def test_rulonnym_does_not_collapse_into_stalnym(index):
    candidates = match("Кровельщик по рулонным кровлям", index, k=2)
    assert candidates[0].code == "КЛС-011"
    assert candidates[0].score > candidates[1].score

    candidates = match("Кровельщик по стальным кровлям", index, k=2)
    assert candidates[0].code == "КЛС-012"
    assert candidates[0].score > candidates[1].score


def test_gidro_does_not_collapse_into_termoizolyatsiya(index):
    candidates = match("Изолировщик на гидроизоляции", index, k=2)
    assert candidates[0].code == "КЛС-013"
    assert candidates[0].score > candidates[1].score

    candidates = match("Изолировщик на термоизоляции", index, k=2)
    assert candidates[0].code == "КЛС-014"
    assert candidates[0].score > candidates[1].score


def test_mashinist_krana_bashennogo_matches_klc_026(index):
    candidates = match("Машинист крана башенного", index, k=1)
    assert candidates[0].code == "КЛС-026"


def test_marketolog_scores_below_t_match(index):
    candidates = match("Маркетолог", index, k=1)
    assert candidates[0].score < 0.55


# --------------------------------------------------------------------------
# Structural properties
# --------------------------------------------------------------------------


def test_build_index_has_56_entries(index):
    assert len(index) == 56


def test_match_returns_k_candidates_sorted_descending(index):
    candidates = match("Плотник", index, k=5)
    assert len(candidates) == 5
    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_match_scores_are_in_unit_range(index):
    for title in ["Плотник", "Маркетолог", "Машинист крана башенного"]:
        for candidate in match(title, index, k=5):
            assert 0.0 <= candidate.score <= 1.0


def test_match_default_k_is_at_least_two(index):
    candidates = match("Плотник", index)
    assert len(candidates) >= 2
