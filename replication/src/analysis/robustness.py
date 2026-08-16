"""
robustness.py — Robustness checks (Table A2, A3).

  - Alternative factor pricing models (CAPM / FF3 / FF5)
  - Alternative e-value tuning / economic-significance deltas
  - Alternative train/test splits
  - Microcap exclusion
"""
from __future__ import annotations

import pandas as pd

from .oos_horse_race import run_oos_horse_race


def run_robustness(factor_data, ff, evalue_results, config) -> dict:
    """Run robustness battery; returns dict of results frames."""
    out = {}
    out["splits"] = {}
    for split in config["empirical"]["alt_splits"]:
        ts, te, vs, ve = split
        # copy config and override split
        cfg = _clone(config)
        cfg["empirical"].update({
            "train_start": ts, "train_end": te,
            "test_start": vs, "test_end": ve,
        })
        df, _ = run_oos_horse_race(factor_data, ff, evalue_results, cfg)
        out["splits"][f"{ts}-{ve}"] = df
    return out


def _clone(d: dict) -> dict:
    import copy
    return copy.deepcopy(d)
