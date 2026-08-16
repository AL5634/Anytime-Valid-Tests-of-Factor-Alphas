"""
ebh.py — e-BH: false-discovery-rate control with e-values
(Wang & Ramdas, 2022, JRSS-B).

Given K e-values for K null hypotheses, the e-BH procedure orders them
descending, e_(1) >= ... >= e_(K), finds

    k_hat = max { k in 1..K : e_(k) >= K / (alpha * k) },

and rejects every hypothesis whose e-value is at least K / (alpha * k_hat).
The procedure controls the false discovery rate at level alpha under any
dependence structure between the e-values.

We use it to lift the per-predictor anytime-valid rejections of
Section 4.1 to a zoo-wide FDR statement over the 212 CZ predictors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ebh_threshold(evalues, alpha: float = 0.05) -> tuple[float, int]:
    """Return (threshold, number of rejections) of the e-BH procedure.

    e-values must be nonnegative; NaN entries are dropped before the
    procedure. If no rejection is possible the threshold is +inf.
    """
    e = np.asarray(evalues, dtype=float)
    e = e[np.isfinite(e)]
    e = e[e >= 0.0]
    e = np.sort(e)[::-1]
    K = len(e)
    if K == 0:
        return float("inf"), 0
    k_hat = 0
    for k in range(1, K + 1):
        if e[k - 1] >= K / (alpha * k):
            k_hat = k
    if k_hat == 0:
        return float("inf"), 0
    thr = K / (alpha * k_hat)
    n_rej = int((e >= thr).sum())
    return float(thr), n_rej


def ebh_rejection_set(evalues: pd.Series, alpha: float = 0.05) -> set:
    """Index set of hypotheses rejected by e-BH at level alpha."""
    thr, _ = ebh_threshold(evalues.to_numpy(), alpha=alpha)
    return set(evalues[evalues >= thr].dropna().index)
