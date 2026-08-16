"""
sensitivity.py — Sensitivity of the e-value rejection set to the two tuning
choices (min_obs, clip_scale).

The paper reports a compressed 4x4 grid of the number of rejected factors
and the robust "core" that survives every cell with clip_scale in {2,3}.
For each (min_obs, clip_scale) cell we recompute the full 212-factor
e-value batch under the FF5 conditioning set and count rejections at
E >= 1/alpha.
"""
from __future__ import annotations

import pandas as pd

from ..evalue.batch import compute_all_factor_evals


def run_sensitivity_grid(
    factor_data: pd.DataFrame,
    ff_factors: pd.DataFrame,
    zoo: list[str],
    min_obs_grid: tuple[int, ...] = (24, 36, 60, 120),
    clip_grid: tuple[float, ...] = (2.0, 3.0, 4.0, 6.0),
    alpha: float = 0.05,
    n_jobs: int = -1,
    summary_factors: list[str] | None = None,
) -> pd.DataFrame:
    """Full 4x4 sensitivity grid. Returns one row per cell with n_rej, the
    comma-joined rejection set, and the final e-values of `summary_factors`
    (default: the six strongest baseline rejectors)."""
    threshold = 1.0 / alpha
    if summary_factors is None:
        base = compute_all_factor_evals(
            factor_data[zoo], ff_factors, n_jobs=n_jobs,
            min_obs=60, alpha=alpha, clip_scale=3.0)
        summary_factors = base.sort_values(
            "evalue_final", ascending=False).head(6).index.tolist()

    rows = []
    for m in min_obs_grid:
        for c in clip_grid:
            ev = compute_all_factor_evals(
                factor_data[zoo], ff_factors, n_jobs=n_jobs,
                min_obs=m, alpha=alpha, clip_scale=c)
            rej = ev[ev["evalue_final"] >= threshold].index.tolist()
            row = {
                "min_obs": m,
                "clip_scale": c,
                "n_rej": len(rej),
                "rejected": ",".join(sorted(rej)),
            }
            for f in summary_factors:
                row[f"E_{f}"] = float(ev.loc[f, "evalue_final"]) if f in ev.index else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)
