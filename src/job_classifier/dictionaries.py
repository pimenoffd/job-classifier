"""Rule data for the normalization pipeline: regexes, abbreviations, synonyms.

This module holds *data only* — `normalize.py` owns the order in which the
rules are applied.  Public interface (used by `normalize.py` and, for the
classifier index, by `matcher.py`):

    load_classifier(path=None) -> list[tuple[str, str]]
        The 56 `(code, canonical name)` pairs read from `data/classifier.csv`.

    load_pattern_rules(path) -> list[tuple[str, str]]
        `pattern;replacement` pairs from a text file, in file order. Backs
        `ABBREVIATION_RULES`, `CRANE_TECHNIQUE_RULES`, `SYNONYM_RULES` below —
        each has its rule data in a `data/*.txt` file, not in this module, so
        the wordlists can be edited without touching code.

    DATA_DIR, CLASSIFIER_PATH
        Default location of the input data, anchored to the repo root via
        `__file__` (not to the current working directory).

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

#: Every default input path is anchored here, not to the current working
#: directory, so the CLI and the tests behave the same from any cwd.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

CLASSIFIER_PATH = DATA_DIR / "classifier.csv"


def load_classifier(path: Path | None = None) -> list[tuple[str, str]]:
    """Read `classifier.csv` into `(code, canonical name)` pairs."""
    with open(path or CLASSIFIER_PATH, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader, None)
        if header is None:
            return []
        return [(row[0].strip(), row[1].strip()) for row in reader if len(row) >= 2 and row[0].strip()]


def load_pattern_rules(path: Path) -> list[tuple[str, str]]:
    """Read `pattern;replacement` pairs from `path`, one per line, in file
    order (application order matters for the rule lists below). Blank lines
    and `#` comments are ignored."""
    rules: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\n").strip()
            if not stripped or stripped.startswith("#"):
                continue
            pattern, _, replacement = line.rstrip("\n").partition(";")
            rules.append((pattern, replacement))
    return rules


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
# Data lives in `data/abbreviations.txt` (`Эл.газосварщик`, `ЭЛ.ГАЗОСВАРЩИК`,
# `Эл.газосврщик` -> электрогазосварщик; `Сварщик ЭГС` is the whole job
# title, not сварщик + something; etc.) — see that file for the full list.
#
# Deliberately absent: a rule splitting the productive prefix `строй` off a
# glued noun (`СТРОЙУАСТКА`, id 38).  Measured: expanding it to
# `мастер строительного участка` makes the record match КЛС-009 Маляр
# строительный at 0.78 — confidently wrong — whereas leaving `стройуастка`
# as an unmatched token keeps КЛС-047 Мастер строительных и монтажных работ
# on top.  See the task-2 report.

_ABBREVIATIONS: list[tuple[str, str]] = load_pattern_rules(DATA_DIR / "abbreviations.txt")
ABBREVIATION_RULES = [(re.compile(p), r) for p, r in _ABBREVIATIONS]


# --------------------------------------------------------------------------
# Stage 7a: technique before role (PLAN.md §4.2)
# --------------------------------------------------------------------------
# The classifier has three cranes (КЛС-025/026/027).  Collapsing every crane
# role to `машинист крана` erases the one word that tells them apart, so the
# *technique* is normalized to a canonical form first, and only then is the
# role word collapsed.

# Data lives in `data/crane_technique.txt` (patterns pre-expanded from the
# inflected-forms-of-`кран` fragment `кран(?:а|у|е|ом|ы|ов|ам|ами|ах)?` —
# `кран[а-я]*` alone would also swallow `крановщик`, which is the *role*,
# not the technique).

_CRANE_TECHNIQUE: list[tuple[str, str]] = load_pattern_rules(DATA_DIR / "crane_technique.txt")
CRANE_TECHNIQUE_RULES = [(re.compile(p), r) for p, r in _CRANE_TECHNIQUE]

# True only once a technique rule has produced the bare canonical `кран`;
# `крановщик` alone does not match (no word boundary after `кран`).
CRANE_TECHNIQUE_MARKER_RE = re.compile(r"\bкран\b")

# Role words that mean "machine operator" once the technique is already known.
CRANE_ROLE_RE = re.compile(r"\b(?:крановщик|оператор)[а-я]*\b")


# --------------------------------------------------------------------------
# Stage 7b: professional synonyms
# --------------------------------------------------------------------------

# Data lives in `data/synonyms.txt` (прораб, сметчик, бульдозерист ->
# машинист бульдозера, крановщик as a fallback for a crane role with no
# technique word at all, оператор экскаватора -> машинист экскаватора, etc.).

_SYNONYMS: list[tuple[str, str]] = load_pattern_rules(DATA_DIR / "synonyms.txt")
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
