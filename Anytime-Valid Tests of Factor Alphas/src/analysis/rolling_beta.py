"""
rolling_beta.py — Rolling-window robustness of the e-value rejection set.

Recomputes the anytime-valid e-process for all 212 CZ predictors with the
loadings estimated on a trailing 120-month window (instead of the expanding
window of the main specification). All other hyperparameters match the
baseline: min_obs=60, clip_scale=3, alpha=0.05, FF5 conditioning.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evalue.core import regression_alpha_evalue

FF_NAMES = ["mkt", "smb", "hml", "rmw", "cma"]
WINDOW = 120


def main() -> None:
    base = Path(__file__).resolve().parents[2]
    univ = pd.read_csv(base / "data/processed/factor_universe.csv",
                       index_col=0, parse_dates=True)
    ff = univ[[c for c in FF_NAMES if c in univ.columns]]
    zoo = [c for c in univ.columns if c not in FF_NAMES]

    rows = {}
    for f in zoo:
        r = regression_alpha_evalue(univ[f], ff, min_obs=60, clip_scale=3.0,
                                    alpha=0.05, beta_window=WINDOW)
        rows[f] = {"evalue_final": r["evalue_final"],
                   "first_rejection_date": r["first_rejection_date"]}

    res = pd.DataFrame(rows).T
    res.index.name = "factor_name"
    out = base / "data/results/evalue_rolling120.csv"
    res.to_csv(out)
    n_rej = int((res["evalue_final"] >= 20).sum())
    print(f"rolling-120: {n_rej}/{len(res)} reject at E>=20")


if __name__ == "__main__":
    main()
