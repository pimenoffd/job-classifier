"""Fuzzy scoring of a job title against the 56-entry classifier index.

Public interface:

    build_index(path=None, vocabulary=None) -> ClassifierIndex
        Normalize every `(code, name)` pair from `data/classifier.csv` once.
        Reuse the returned index across queries instead of rebuilding it.

    match(query, index, k=5, vocabulary=None) -> list[Candidate]
        Score `query` against every entry in `index` and return the top `k`
        candidates, sorted descending by score.

Score formula (controller ruling, task-3-brief.md): `w *
rapidfuzz.fuzz.token_sort_ratio + (1 - w) * rapidfuzz.fuzz.ratio`, with `w`
= `config.toml`'s `matcher.token_sort_ratio_weight` (shipped at 0.5),
computed on the two sides' `normalize()` output and scaled from rapidfuzz's
0-100 range to `[0, 1]` — matching the scale of `docs/PLAN.md`'s threshold
constants (`T_match = 0.55` etc.). This overrides §7 step 3's literal
`WRatio` wording: PLAN.md §3's calibration experiment that produced
`T_match` was run against `ratio`, not `WRatio`, and `WRatio`'s
partial-token matching was measured to let through 6x more office-role
false positives at the same threshold (see task-3-brief.md for the full
account).

This module only scores and ranks candidates. It does not apply
`T_match`/`T_confident`/`T_margin` or decide match/no-match — that's
`decision.py` (Task 4).
"""

from __future__ import annotations

from typing import NamedTuple

from rapidfuzz import fuzz

from .config import CONFIG
from .dictionaries import load_classifier
from .normalize import normalize

#: Number of candidates `match()` returns by default, from `config.toml`
#: (`[matcher]` table). `decision.py` (Task 4) needs at least the top 2 for
#: its margin check; the shipped value of 5 is headroom beyond that.
DEFAULT_K = CONFIG.matcher.default_k


class IndexEntry(NamedTuple):
    code: str
    name: str
    normalized_name: str


#: `build_index()`'s return type: one normalized entry per classifier row.
ClassifierIndex = tuple[IndexEntry, ...]


class Candidate(NamedTuple):
    code: str
    name: str
    score: float


def build_index(path=None, vocabulary: tuple[str, ...] | None = None) -> ClassifierIndex:
    """Normalize the 56 classifier entries once, ahead of any query.

    `path` is forwarded to `load_classifier()`; `vocabulary` is forwarded to
    `normalize()` (default: `normalize.default_vocabulary()`).
    """
    return tuple(
        IndexEntry(code, name, normalize(name, vocabulary)) for code, name in load_classifier(path)
    )


def _score(query_normalized: str, entry_normalized: str) -> float:
    weight = CONFIG.matcher.token_sort_ratio_weight
    token_sort = fuzz.token_sort_ratio(query_normalized, entry_normalized)
    ratio = fuzz.ratio(query_normalized, entry_normalized)
    return (weight * token_sort + (1 - weight) * ratio) / 100


def match(
    query: str,
    index: ClassifierIndex,
    k: int = DEFAULT_K,
    vocabulary: tuple[str, ...] | None = None,
) -> list[Candidate]:
    """Score `query` against every entry in `index`, return the top `k`."""
    query_normalized = normalize(query, vocabulary)
    scored = [
        Candidate(entry.code, entry.name, _score(query_normalized, entry.normalized_name))
        for entry in index
    ]
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:k]
