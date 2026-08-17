"""Rule data for the normalization pipeline: regexes, abbreviations, synonyms.

This module holds *data only* — `normalize.py` owns the order in which the
rules are applied.  Public interface (used by `normalize.py` and, for the
classifier index, by `matcher.py`):

    load_classifier(path=None) -> list[tuple[str, str]]
        The 56 `(code, canonical name)` pairs read from `data/classifier.csv`.

    CLASSIFIER_PATH
        Default location of that file.

    COMPANY_RE, STATUS_TAG_RE, GRADE_RE
        Garbage-tail removal (PLAN.md §2 "мусорные хвосты").

    ABBREVIATION_RULES, CRANE_TECHNIQUE_RULES, CRANE_ROLE_RE,
    CRANE_TECHNIQUE_MARKER_RE, SYNONYM_RULES
        Ordered lists of `(compiled regex, replacement)` substitutions.

    rule_vocabulary() -> set[str]
        Every Cyrillic token the rules depend on.  Together with the
        classifier tokens this is the correction dictionary of PLAN.md §4.1.

Every rule below is grounded in a form that actually occurs in
`data/raw_positions.csv`; the docs' seed examples that also occur there are
kept, the ones that do not occur are noted in the task report.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

CLASSIFIER_PATH = Path(__file__).resolve().parents[2] / "data" / "classifier.csv"


def load_classifier(path: Path | None = None) -> list[tuple[str, str]]:
    """Read `classifier.csv` into `(code, canonical name)` pairs."""
    with open(path or CLASSIFIER_PATH, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader, None)
        if header is None:
            return []
        return [(row[0].strip(), row[1].strip()) for row in reader if len(row) >= 2 and row[0].strip()]


# --------------------------------------------------------------------------
# Stage 2-4: garbage tails
# --------------------------------------------------------------------------

# `ООО "СтройМонтаж"`, `АО "МостСтрой"` — org form followed by a quoted name.
COMPANY_RE = re.compile(r"\b(?:ооо|оао|зао|пао|ао|ип|нко)\b\s*[«\"'][^»\"']*[»\"']")

# Employment-status tags in parentheses.  Deliberately an explicit list: the
# data also contains `Прораб (производитель работ)`, where the parenthesized
# text is content and must survive.
STATUS_TAG_RE = re.compile(r"\(\s*(?:осн|совм|вахта|врем|подмен|сокр)\s*\.?\s*\)")

# Verbatim from docs/proposed architecture..md §3.1.
GRADE_RE = re.compile(r"\b\d+[- ]*(й|го)?\s*(разряд[а-я]*|кат[а-я]*|класс[а-я]*)")


# --------------------------------------------------------------------------
# Stage 5: domain abbreviations
# --------------------------------------------------------------------------
# Applied to the raw (still punctuated) string, because the trailing dot is
# what makes most of these unambiguous.  Order matters: longer keys first, so
# `металлоконстр.` is not eaten by `констр.` (the \b guard also prevents it).

_ABBREVIATIONS: list[tuple[str, str]] = [
    # `Эл.газосварщик`, `ЭЛ.ГАЗОСВАРЩИК`, `Эл.газосврщик` -> электрогазосварщик
    (r"\bэл\.\s*", "электро"),
    # `Сварщик ЭГС` is the whole job title, not сварщик + something.
    (r"\bсварщик\s+эгс\b", "электрогазосварщик"),
    (r"\bэгс\b", "электрогазосварщик"),
    (r"\bпто\b", "производственно-технического отдела"),
    (r"\bсмр\b", "строительных и монтажных работ"),
    (r"\bмаш\.", "машинист"),
    (r"\bнач\.", "начальник"),
    (r"\bмех\.", "механического"),
    (r"\bруч\.", "ручного"),
    (r"\bметаллоконстр\.", "металлоконструкций"),
    (r"\bконстр\.", "конструкций"),
    (r"\bсантех\.", "санитарно-технических"),
    (r"\bтехнолог\.", "технологических"),
    (r"\bавтомоб\.", "автомобильного"),
    (r"\bстроит\.", "строительных"),
]
# Deliberately absent: a rule splitting the productive prefix `строй` off a
# glued noun (`СТРОЙУАСТКА`, id 38).  Measured: expanding it to
# `мастер строительного участка` makes the record match КЛС-009 Маляр
# строительный at 0.78 — confidently wrong — whereas leaving `стройуастка`
# as an unmatched token keeps КЛС-047 Мастер строительных и монтажных работ
# on top.  See the task-2 report.

ABBREVIATION_RULES = [(re.compile(p), r) for p, r in _ABBREVIATIONS]


# --------------------------------------------------------------------------
# Stage 7a: technique before role (PLAN.md §4.2)
# --------------------------------------------------------------------------
# The classifier has three cranes (КЛС-025/026/027).  Collapsing every crane
# role to `машинист крана` erases the one word that tells them apart, so the
# *technique* is normalized to a canonical form first, and only then is the
# role word collapsed.

# Inflected forms of the noun `кран` only — `кран[а-я]*` would also swallow
# `крановщик`, which is the *role*, not the technique.
_KRAN = r"кран(?:а|у|е|ом|ы|ов|ам|ами|ах)?"

_CRANE_TECHNIQUE: list[tuple[str, str]] = [
    (r"\bавтокран[а-я]*\b", "кран автомобильный"),
    (rf"\b(?:башенн[а-я]*\s+{_KRAN}|{_KRAN}\s+башенн[а-я]*)\b", "кран башенный"),
    (rf"\b(?:гусеничн[а-я]*\s+{_KRAN}|{_KRAN}\s+гусеничн[а-я]*)\b", "кран гусеничный"),
    (rf"\b(?:автомобильн[а-я]*\s+{_KRAN}|{_KRAN}\s+автомобильн[а-я]*)\b", "кран автомобильный"),
]

CRANE_TECHNIQUE_RULES = [(re.compile(p), r) for p, r in _CRANE_TECHNIQUE]

# True only once a technique rule has produced the bare canonical `кран`;
# `крановщик` alone does not match (no word boundary after `кран`).
CRANE_TECHNIQUE_MARKER_RE = re.compile(r"\bкран\b")

# Role words that mean "machine operator" once the technique is already known.
CRANE_ROLE_RE = re.compile(r"\b(?:крановщик|оператор)[а-я]*\b")


# --------------------------------------------------------------------------
# Stage 7b: professional synonyms
# --------------------------------------------------------------------------

_SYNONYMS: list[tuple[str, str]] = [
    (r"\bпроизводител[а-я]*\s+работ[а-я]*\b", "прораб"),
    (r"\bспециалист[а-я]*\s+по\s+сметам\b", "сметчик"),
    (r"\bбульдозерист[а-я]*\b", "машинист бульдозера"),
    (r"\bразнорабоч[а-я]*\b", "подсобный рабочий"),
    (r"\bшофер[а-я]*\b", "водитель"),
    # Fallback for a crane role with no technique word at all.
    (r"\bкрановщик[а-я]*\b", "машинист крана"),
    # `операттор экскаватора` -> `машинист экскаватора`.
    (r"\bоператор[а-я]*\b", "машинист"),
]

SYNONYM_RULES = [(re.compile(p), r) for p, r in _SYNONYMS]


# --------------------------------------------------------------------------
# Correction vocabulary (PLAN.md §4.1)
# --------------------------------------------------------------------------

_CYRILLIC_RUN = re.compile(r"[а-я]{2,}")


def rule_vocabulary() -> set[str]:
    """Cyrillic tokens the rules depend on, harvested from the rules themselves.

    Both sides of the synonym / crane rules are included: a query token must be
    able to snap onto a *pattern* token (so `произвдоитель` -> `производител`
    lets the phrase rule fire) as well as onto a *replacement* token.
    Abbreviation patterns are deliberately excluded — they are the noisy forms
    the pipeline expands away before correction runs, and their short stubs
    (`маш`, `нач`, `эл`) would be dangerous correction targets.
    """
    tokens: set[str] = set()
    for pattern, replacement in _CRANE_TECHNIQUE + _SYNONYMS:
        tokens.update(_CYRILLIC_RUN.findall(pattern))
        tokens.update(_CYRILLIC_RUN.findall(replacement))
    for _pattern, replacement in _ABBREVIATIONS:
        tokens.update(_CYRILLIC_RUN.findall(replacement))
    return tokens
