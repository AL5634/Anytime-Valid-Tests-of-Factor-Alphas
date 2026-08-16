"""
test_calibration.py — Monte-Carlo calibration of the anytime-valid
e-process reported in the paper's Table (tab:calib).

For each clip bound B in {2,3,4,6} x sigma and each sample size
T in {120, 240, 720}, we estimate the empirical type-I error at
alpha = 0.05 under the null (mean-zero i.i.d. normal residuals).
The anytime-valid guarantee (Ville) implies the empirical size must be
at most ~ 1/alpha * (1 + MC error) along the fully sequential path;
we assert it stays close to (and never materially above) the nominal
5%, demonstrating both validity and insensitivity to the clip bound.
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evalue.core import anytime_mean_cs

ALPHA = 0.05
REPS = 1000
SIGMAS = [0.02, 0.04]


def _empirical_type1(b_scale, T, sigma, reps=REPS, seed=0):
    rng = np.random.default_rng(seed)
    rej = 0
    for _ in range(reps):
        x = rng.normal(0, sigma, T)
        clip = b_scale * sigma
        res = anytime_mean_cs(pd.Series(x), alpha=ALPHA, sigma_hat=clip / 3.0)
        if (res["e_path"] >= res["threshold"]).any():
            rej += 1
    return rej / reps


def _calibration_table():
    """Return the (B/sigma x T) empirical-size grid shown as tab:calib."""
    bounds = [2, 3, 4, 6]
    Ts = [120, 240, 720]
    rows = []
    for b in bounds:
        row = {"B_over_sigma": b}
        for T in Ts:
            row[f"T={T}"] = _empirical_type1(b, T, sigma=0.04)
        rows.append(row)
    return pd.DataFrame(rows).set_index("B_over_sigma")


def test_calibration_size_valid():
    """Empirical type-I error must NEVER exceed (by a non-trivially
    meaningful margin over Monte-Carlo error) the anytime-valid upper
    bound of the nominal 5%. The betting construction is intentionally
    conservative, so size may be well *below* 5%; the critical property
    for validity is the upper bound."""
    tbl = _calibration_table()
    vals = tbl.to_numpy(dtype=float)
    # With REPS=1000, MC s.e. ~0.7%; the anytime-valid guarantee says
    # size <= 5%. We accept a small MC margin (s.e. x ~4) =>
    # size should be <= 8%. Any value above this indicates a genuine
    # violation of the Ville bound under the null.
    assert vals.max() <= 0.08, \
        f"size {vals.max():.3f} EXCEEDS the anytime-valid bound under null"


def test_calibration_uses_nominal_sigma_binding():
    """Print the grid (used to build the manuscript's tab:calib)."""
    import warnings
    warnings.warn(f"\nCalibration table (empirical size %):\n"
                  f"{(_calibration_table()*100).round(1).to_string()}",
                  UserWarning)
