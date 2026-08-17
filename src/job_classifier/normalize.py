"""Normalization pipeline: noisy 1C job title -> canonical comparable string.

Public interface (this is what `matcher.py` calls):

    normalize(text) -> str
        Full pipeline.  Apply it to *both* sides — raw titles and the 56
        classifier names — and compare the results.  Two spellings of the same
        job are expected to produce byte-identical strings; scoring in
        `matcher.py` then only has to deal with genuinely different wordings.

    default_vocabulary() -> tuple[str, ...]
        The correction dictionary of PLAN.md §4.1: classifier tokens + tokens
        the rules in `dictionaries.py` depend on.  Cached; pass a different
        tuple as `normalize(text, vocabulary=...)` to override (tests).

Pipeline order (docs/proposed architecture..md §3.1, with PLAN.md §4.1/§4.2
inserted as stages 6 and 7a):

    1. lowercase, ё -> е, strip
    2. drop `ООО "…"` / `АО "…"` company names
    3. drop `(осн.)` / `(совм.)` / `(вахта)` status tags
    4. drop `5 разряда` / `1 категории` / `2 класса` tails
    5. expand abbreviations (`маш.`, `руч.`, `ПТО`, `ЭГС`, `Эл.` …)
    6. token correction against the vocabulary (fixes typos *before* the
       multi-word phrase rules try to match — PLAN.md §4.1)
    7. a) crane technique -> canonical form, then role word -> `машинист`
          (PLAN.md §4.2); b) remaining professional synonyms
    8. Snowball-RU stemming, duplicate-token removal, whitespace collapse

Stage 8 removes duplicate stems so that `Прораб (производитель работ)`
(-> `прораб прораб`) and `Сварщик ЭГС` do not end up with doubled tokens.  It
is applied identically to both sides, so it cannot skew a comparison.
"""

from __future__ import annotations

import functools
import re

import snowballstemmer
from rapidfuzz import fuzz, process

from . import dictionaries as rules

_STEMMER = snowballstemmer.stemmer("russian")

TOKEN_RE = re.compile(r"[a-zа-я0-9]+")

#: rapidfuzz scores are 0-100, so PLAN.md §4.1's "ratio >= 0.80" is 80 here.
CORRECTION_SCORE_CUTOFF = 80.0
#: A typo changes a word's length by at most a character or two.  Without this
#: window `делопроизводитель` scores 83 against the rule token `производител`
#: and would be silently rewritten into a construction term.
CORRECTION_MAX_LENGTH_DELTA = 2
#: Shorter tokens are left alone; at 1-2 characters `ratio` is pure noise.
CORRECTION_MIN_TOKEN_LENGTH = 3


@functools.lru_cache(maxsize=1)
def default_vocabulary() -> tuple[str, ...]:
    """Correction dictionary: classifier tokens + tokens used by the rules."""
    tokens: set[str] = set(rules.rule_vocabulary())
    for _code, name in rules.load_classifier():
        tokens.update(TOKEN_RE.findall(name.lower().replace("ё", "е")))
    return tuple(sorted(tokens))


@functools.lru_cache(maxsize=8)
def _correction_index(vocabulary: tuple[str, ...]) -> tuple[frozenset[str], dict[int, list[str]]]:
    """Membership set + candidates bucketed by token length."""
    by_length: dict[int, list[str]] = {}
    for token in vocabulary:
        by_length.setdefault(len(token), []).append(token)
    return frozenset(vocabulary), by_length


@functools.lru_cache(maxsize=4096)
def correct_token(token: str, vocabulary: tuple[str, ...]) -> str:
    """Snap an out-of-vocabulary token onto its closest dictionary entry.

    Returns `token` unchanged when it is already known, too short, contains a
    digit, or has no neighbour scoring `>= CORRECTION_SCORE_CUTOFF` within the
    length window.
    """
    known, by_length = _correction_index(vocabulary)
    if token in known or len(token) < CORRECTION_MIN_TOKEN_LENGTH or not token.isalpha():
        return token

    candidates: list[str] = []
    for length in range(
        len(token) - CORRECTION_MAX_LENGTH_DELTA, len(token) + CORRECTION_MAX_LENGTH_DELTA + 1
    ):
        candidates.extend(by_length.get(length, ()))
    if not candidates:
        return token

    best = process.extractOne(
        token, candidates, scorer=fuzz.ratio, score_cutoff=CORRECTION_SCORE_CUTOFF
    )
    return best[0] if best else token


def normalize(text: str, vocabulary: tuple[str, ...] | None = None) -> str:
    """Normalize one job title.  See the module docstring for the stage list."""
    vocabulary = default_vocabulary() if vocabulary is None else vocabulary

    # 1-4: cleanup and garbage tails.
    s = text.lower().replace("ё", "е").strip()
    s = rules.COMPANY_RE.sub(" ", s)
    s = rules.STATUS_TAG_RE.sub(" ", s)
    s = rules.GRADE_RE.sub(" ", s)

    # 5: abbreviations (still on the punctuated string — the dots matter).
    for pattern, replacement in rules.ABBREVIATION_RULES:
        s = pattern.sub(replacement, s)

    # 6: typo correction, so the phrase rules below see clean tokens.
    tokens = [correct_token(t, vocabulary) for t in TOKEN_RE.findall(s)]
    s = " ".join(tokens)

    # 7a: technique first, role second (PLAN.md §4.2).
    for pattern, replacement in rules.CRANE_TECHNIQUE_RULES:
        s = pattern.sub(replacement, s)
    if rules.CRANE_TECHNIQUE_MARKER_RE.search(s):
        s = rules.CRANE_ROLE_RE.sub("машинист", s)

    # 7b: remaining professional synonyms.
    for pattern, replacement in rules.SYNONYM_RULES:
        s = pattern.sub(replacement, s)

    # 8: stem, drop duplicate stems, collapse whitespace.
    stems = _STEMMER.stemWords(s.split())
    return " ".join(dict.fromkeys(stems))
