"""Decision layer: turn scored candidates into a code / confidence / review flag.

Public interface:

    make_decision(top1_code, top1_name, s1, s2, normalized_query=None,
                  threshold_match=THRESHOLD_MATCH) -> Decision
        The full decision rule of docs/proposed architecture..md §3.3 with
        PLAN.md §4.3's calibrated OOD confidence applied.

    decide(candidates, normalized_query=None, threshold_match=THRESHOLD_MATCH)
        Convenience wrapper over `matcher.match()`'s return value.  Pass
        `normalized_query` — omitting it silently disables the safeguard.

    is_out_of_scope(normalized_query) -> bool
        The out-of-scope safeguard (see below).

Thresholds are **fixed constants**, calibrated on data in PLAN.md §3 — not
re-derived at run time.  `threshold_match` is a parameter only so that
`evaluate`'s diagnostic sweep can print metrics at neighbouring values; the
shipped pipeline always uses the constant.

Confidence semantics (PLAN.md §4.3): the number is *confidence in the
decision that was taken*, on one scale in both branches.  For a match it is
the similarity score; for a rejection it is how far below the acceptance
boundary the best candidate fell, normalised by the empirical floor of
non-construction scores.  `1.0 - s1` (the original document's formula) is
deliberately not used: it mixes "confidence in the code" with "confidence in
the rejection" and peaks at exactly the point of maximum uncertainty.

Out-of-scope safeguard (task-4-brief.md ruling 1): lexical similarity cannot
separate every non-construction title from the classifier.  Measured leak:
`Переводчик` scores 0.737 against `КЛС-056 Проходчик` — a pure
character-level coincidence (7 of 10 letters shared, in order) that no
normalization rule or metric choice removes.  PLAN.md §9 requires all 20
known office roles to come out as `НЕТ СООТВЕТСТВИЯ`, so a curated stem list
overrides the score for them — kept in `data/out_of_scope_stems.txt`,
loaded via `load_out_of_scope_stems()`. `config.toml`'s
`decision.out_of_scope_safeguard_enabled` ships **off**: measured that way,
`THRESHOLD_CONFIDENT` already keeps the one scoring leak (`Переводчик`) out
of the auto-accepted branch on its own, so disabling the curated list costs
review-queue size (4 -> 23 of 300 rows, all correctness-neutral: no rejection
that skips review ever fires without the list, per `make_decision` step 1),
not correctness — see NOTE.md. Flip that config key to `true` to shrink the
review queue back down.

**Known limitation, to be disclosed in NOTE.md:** this list covers
*previously observed* non-construction vocabulary only.  It is not an
open-ended office-role detector; a new out-of-scope title whose spelling
happens to collide with a classifier entry would still slip through to the
"match" branch (where the confidence threshold would flag it for review, so
it is not silently auto-accepted).

The safeguard also has to yield when the *same* title carries both signals —
`Мастер строительных и монтажных работ / менеджер` is a genuine construction
role with an office word appended.  A curated hit on such a title is no
longer a certain rejection, so it is routed to a human instead of being
silently discarded; see `make_decision` step 0.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from rapidfuzz import fuzz

from .config import CONFIG
from .dictionaries import DATA_DIR
from .matcher import Candidate

#: Tunables below are read from `config.toml` (`[decision]` table) — see
#: `config.py`. PLAN.md §3: max(OOD) = 0.541, min(match) = 0.632; 8% review
#: load at `threshold_confident`; PLAN.md §4.3 for `s_floor`'s empirical
#: floor (`Казначей` 0.375, `Юрисконсульт` 0.385, `психолог` 0.400).
THRESHOLD_MATCH = CONFIG.decision.threshold_match
THRESHOLD_CONFIDENT = CONFIG.decision.threshold_confident
THRESHOLD_MARGIN = CONFIG.decision.threshold_margin
S_FLOOR = CONFIG.decision.s_floor
#: When `True` (shipped default), every `НЕТ СООТВЕТСТВИЯ` rejection
#: requires review regardless of score. When `False`, only rejections within
#: `REVIEW_BAND` of `threshold_match` do — see `make_decision` step 1.
REVIEW_ALL_REJECTIONS = CONFIG.decision.review_all_rejections
REVIEW_BAND = CONFIG.decision.review_band

#: Literal sentinel from the task spec — a marker, not a code.
NO_MATCH = "НЕТ СООТВЕТСТВИЯ"

#: The `requires_review` column is Russian text in the deliverable CSV.
REVIEW_YES = "да"
REVIEW_NO = "нет"


class Decision(NamedTuple):
    code: str
    name: str
    confidence: float
    requires_review: str


# ---------------------------------------------------------------------------
# Out-of-scope safeguard
# ---------------------------------------------------------------------------

#: The curated stem list, as a plain text file (one stem per line, `#`
#: comments ignored) — see `load_out_of_scope_stems()`.  Stems, because
#: `normalize()` stems its output: `кадров` -> `кадр`, `агроном` -> `агрон`,
#: `делопроизводитель` -> `делопроизводител`.  `Системный администратор` is
#: keyed on `администратор` — `системн` alone is too generic.  Verified
#: (tests/test_decision.py): fires on none of the 56 classifier names, nor on
#: any of the 280 genuine construction rows of `raw_positions.csv`.
OUT_OF_SCOPE_STEMS_PATH = DATA_DIR / "out_of_scope_stems.txt"


def load_out_of_scope_stems(path: Path) -> tuple[str, ...]:
    """Read one stem per line from `path`. Blank lines and `#` comments ignored."""
    with open(path, encoding="utf-8") as f:
        return tuple(line.strip() for line in f if line.strip() and not line.strip().startswith("#"))


#: Toggled by `config.toml`'s `decision.out_of_scope_safeguard_enabled`
#: (off by default — no file read to express "no curated stems"); flip it in
#: the config file to turn the safeguard back on.
OUT_OF_SCOPE_STEMS: tuple[str, ...] = (
    load_out_of_scope_stems(OUT_OF_SCOPE_STEMS_PATH)
    if CONFIG.decision.out_of_scope_safeguard_enabled
    else ()
)

#: The input is as noisy as everything else here (`Агроом`, `Диспетче`,
#: `Кладощвик` all occur), so the stem lookup is typo-tolerant, mirroring
#: `normalize.correct_token`.  Set tighter than that function's cutoff: a
#: false positive here rejects a real construction worker outright, whereas
#: a correction there only nudges a token.
OUT_OF_SCOPE_SCORE_CUTOFF = CONFIG.decision.out_of_scope_score_cutoff
#: Same guard as `normalize.CORRECTION_MAX_LENGTH_DELTA`: a typo shifts a
#: word's length by a character or two, so `кадр` cannot swallow `кадровщик`.
OUT_OF_SCOPE_MAX_LENGTH_DELTA = CONFIG.decision.out_of_scope_max_length_delta


def is_out_of_scope(normalized_query: str) -> bool:
    """True if a token of `normalized_query` is known non-construction vocabulary.

    `normalized_query` must be `normalize()` output — the stems below are
    stemmed forms, so a raw title will not match.
    """
    for token in normalized_query.split():
        for stem in OUT_OF_SCOPE_STEMS:
            if abs(len(token) - len(stem)) > OUT_OF_SCOPE_MAX_LENGTH_DELTA:
                continue
            if fuzz.ratio(token, stem) >= OUT_OF_SCOPE_SCORE_CUTOFF:
                return True
    return False


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------


def _ood_confidence(s1: float, threshold_match: float) -> float:
    """Confidence *in a rejection*: how far below the boundary the best fell.

    Clamped to [0, 1], so a rejection taken while `s1` is at or above the
    boundary — only the safeguard does that — comes out at 0.0: the score
    offers no support whatsoever for the rejection.
    """
    raw = (threshold_match - s1) / (threshold_match - S_FLOOR)
    return round(min(1.0, max(0.0, raw)), 3)


def make_decision(
    top1_code: str,
    top1_name: str,
    s1: float,
    s2: float,
    normalized_query: str | None = None,
    threshold_match: float = THRESHOLD_MATCH,
) -> Decision:
    """Apply the thresholds to the top-2 scores and return the final verdict.

    `normalized_query` enables the out-of-scope safeguard; when it is `None`
    the decision is made on the scores alone.
    """
    # 0a. An empty title (a blank position field is realistic in a 1C export)
    #     carries no signal at all: every score is 0, which would otherwise
    #     read as a maximally confident rejection.  Reject, but send it to a
    #     human — there is nothing here to be confident about.
    if normalized_query is not None and not normalized_query.strip():
        return Decision(NO_MATCH, "", 0.0, REVIEW_YES)

    # 0b. Curated out-of-scope vocabulary overrides the score.  Two cases:
    #     - no competing signal (`s1` below the acceptance boundary): the hit
    #       is a dictionary fact and the rejection is certain — the common
    #       case, e.g. a bare `Экономист планового отдела`.
    #     - a collision (`s1` at or above the boundary): the title also
    #       matches a real construction role strongly, e.g. `Мастер
    #       строительных и монтажных работ / менеджер` at 0.880.  The two
    #       signals contradict each other, so the rejection is kept (guessing
    #       the "real" code from a mixed title is not the safeguard's job) but
    #       it is no longer certain and must reach a human.
    if normalized_query is not None and is_out_of_scope(normalized_query):
        if s1 < threshold_match:
            return Decision(NO_MATCH, "", 1.0, REVIEW_NO)
        return Decision(NO_MATCH, "", _ood_confidence(s1, threshold_match), REVIEW_YES)

    # 1. Out-of-Distribution: nothing in the classifier is close enough. A low
    #    score is not a verified fact the way a dictionary hit is — it can
    #    also mean the normalization vocabulary didn't recognize a genuine,
    #    unfamiliar construction title. `REVIEW_ALL_REJECTIONS` (shipped
    #    default) sends every rejection to a human regardless of how far
    #    below the boundary it falls; set it to `False` in config.toml to
    #    only review the ones within `REVIEW_BAND` of the boundary and
    #    auto-reject the rest.
    if s1 < threshold_match:
        confidence = _ood_confidence(s1, threshold_match)
        if REVIEW_ALL_REJECTIONS:
            requires_review = REVIEW_YES
        else:
            requires_review = REVIEW_YES if s1 >= (threshold_match - REVIEW_BAND) else REVIEW_NO
        return Decision(NO_MATCH, "", confidence, requires_review)

    # 2. Match found.
    confidence = round(s1, 3)
    margin = s1 - s2
    if confidence < THRESHOLD_CONFIDENT or margin < THRESHOLD_MARGIN:
        requires_review = REVIEW_YES
    else:
        requires_review = REVIEW_NO
    return Decision(top1_code, top1_name, confidence, requires_review)


def decide(
    candidates: list[Candidate],
    normalized_query: str | None = None,
    threshold_match: float = THRESHOLD_MATCH,
) -> Decision:
    """`make_decision` fed straight from `matcher.match()`'s return value.

    Pass `normalize(title)` as `normalized_query`: it is what enables the
    out-of-scope safeguard, and leaving it `None` disables that safeguard
    silently — known non-construction titles would then be decided on score
    alone, and `Переводчик` (0.737 against КЛС-056 Проходчик) would come back
    as a match instead of `НЕТ СООТВЕТСТВИЯ`.
    """
    if not candidates:
        return Decision(NO_MATCH, "", 1.0, REVIEW_NO)
    top1 = candidates[0]
    s2 = candidates[1].score if len(candidates) > 1 else 0.0
    return make_decision(
        top1.code,
        top1.name,
        top1.score,
        s2,
        normalized_query=normalized_query,
        threshold_match=threshold_match,
    )
