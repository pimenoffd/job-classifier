"""Runtime configuration: load `config.toml` into typed, importable constants.

Public interface:

    CONFIG_PATH
        Default location of `config.toml`, anchored to the repo root via
        `__file__` (not to the current working directory) — same convention
        as `dictionaries.DATA_DIR`.

    load_config(path=None) -> Config
        Read and validate `config.toml`.

    CONFIG
        The module-level `Config` loaded from `CONFIG_PATH`. `decision.py`,
        `normalize.py`, and `matcher.py` read their tunables from this at
        import time rather than holding their own hardcoded constants.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import NamedTuple

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.toml"


class DecisionConfig(NamedTuple):
    threshold_match: float
    threshold_confident: float
    threshold_margin: float
    s_floor: float
    out_of_scope_score_cutoff: float
    out_of_scope_max_length_delta: int
    out_of_scope_safeguard_enabled: bool
    review_all_rejections: bool
    review_band: float


class NormalizeConfig(NamedTuple):
    correction_score_cutoff: float
    correction_max_length_delta: int
    correction_min_token_length: int


class MatcherConfig(NamedTuple):
    token_sort_ratio_weight: float
    default_k: int


class Config(NamedTuple):
    decision: DecisionConfig
    normalize: NormalizeConfig
    matcher: MatcherConfig


def load_config(path: Path | None = None) -> Config:
    """Read `config.toml` (or `path`) into a `Config`."""
    with open(path or CONFIG_PATH, "rb") as f:
        raw = tomllib.load(f)
    return Config(
        decision=DecisionConfig(**raw["decision"]),
        normalize=NormalizeConfig(**raw["normalize"]),
        matcher=MatcherConfig(**raw["matcher"]),
    )


CONFIG = load_config()
