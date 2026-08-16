"""
power.py — Simulation-based size and power study for the regression-alpha
e-process (Table 6 in the paper, "Power").

Reports how often the e-process rejects H0: alpha = 0 when a true alpha
is present, and how well Type-I error is controlled when it is not.

Data-generating process mirrors the empirical setting: monthly factor
returns are generated as

    r_t = alpha + beta' FF_t + sigma * eps_t,

where FF_t are the real Fama-French five-factor returns (so the factor
residuals inherit realistic non-normalities and common time-series
features), beta is estimated from a real anomaly (momentum), and sigma is
the anomaly's residual volatility. eps_t is iid N(0,1). Each (alpha,
horizon) cell is simulated B times; we report the frequency of rejection
at any time during the monitoring period (anytime-valid rejection) and of
rejection at the end.

Horizons: the full 1963-2023 sample (726 months) and a 20-year window
(240 months). Alphas: 0, 1, 2, 3, 4, 5 %/yr.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..evalue.core import regression_alpha_evalue

# Candidate momentum factors in the active universe: the Chen-Zimmermann
# acronym for 12-month momentum, or the Ken French name in the KF universe.
# (Used only to identify the factor panel; the DGP is calibrated on the
# MEDIAN residual-volatility factor, not momentum -- see below.)
MOMENTUM_CANDIDATES = ["Mom12m", "mom", "ResidualMomentum"]


def _realistic_components(ff: pd.DataFrame):
    """
    Fit a realistic factor-return DGP from the empirical panel: the DGP is
    calibrated to the MEDIAN residual-volatility predictor in the active
    universe (rather than an idiosyncratic extreme such as momentum, which
    sits in the top decile of residual variance). Returns (beta, sigma).

    beta is the FF5 loading of that factor and sigma its residual sd, so
    the systematic component is removed by the FF5 residualization and the
    e-process sees a mean-shift alpha on iid residuals of realistic scale.
    """
    import statsmodels.api as sm
    import os
    panel_path = os.path.join(os.path.dirname(__file__), "..", "..",
                              "data", "processed", "factor_universe.csv")
    panel = pd.read_csv(panel_path, index_col=0, parse_dates=True)
    zoo = [c for c in panel.columns if c not in ff.columns]
    sigmas = {}
    for f in zoo:
        y = panel[f].dropna()
        X = sm.add_constant(ff.reindex(y.index)).dropna()
        y = y.reindex(X.index)
        if len(y) < 120:
            continue
        sigmas[f] = float(np.sqrt(sm.OLS(y, X).fit().mse_resid))
    if not sigmas:
        raise ValueError("No factors with sufficient history for the power DGP")
    med = pd.Series(sigmas).sort_values()
    f_star = med.index[len(med) // 2]
    y = panel[f_star].dropna()
    X = sm.add_constant(ff.reindex(y.index)).dropna()
    y = y.reindex(X.index)
    res = sm.OLS(y, X).fit()
    return np.asarray(res.params[1:]), float(np.sqrt(res.mse_resid))


def simulate_power(ff: pd.DataFrame, rng: np.random.Generator, B: int = 800,
                   horizons_months: tuple[int, ...] = (240, 726),
                   alphas_annual: tuple[float, ...] = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10),
                   alpha_level: float = 0.05) -> pd.DataFrame:
    """
    Simulated size and power. Returns a tidy DataFrame with columns:
    alpha_annual, horizon_months, reject_anytime, reject_final,
    t_test_power (analytic power of a fixed-sample two-sided 5% t-test on
    the regression alpha, for comparison), n_sim.

    The DGP draws factor returns r_t = alpha + beta' FF_t + sigma*eps_t
    with beta/sigma from the median-volatility predictor in the active
    universe (see _realistic_components), so the systematic part
    is removed by the FF5 residualization and the e-process sees a mean-shift
    alpha on iid residuals.
    """
    beta, sigma = _realistic_components(ff)
    rows = []
    for T in horizons_months:
        # long enough FF history: use the tail of the real FF panel
        ff_window = ff.iloc[-T:].to_numpy()
        for alpha_ann in alphas_annual:
            alpha_m = alpha_ann / 12.0
            rej_any = 0
            rej_final = 0
            for _ in range(B):
                eps = rng.normal(0.0, sigma, T)
                r = alpha_m + ff_window @ beta + eps
                r_series = pd.Series(r, index=ff.index[-T:])
                e = regression_alpha_evalue(
                    r_series, ff.iloc[-T:], min_obs=60,
                    clip_scale=3.0, alpha=alpha_level)["e_path"]
                thr = 1.0 / alpha_level
                rej_any += bool((e >= thr).any())
                rej_final += bool(not e.empty and e.iloc[-1] >= thr)
            # analytic power of the fixed-sample two-sided t-test at the
            # same 5% level: t ~ N(alpha_m*sqrt(T)/sigma, 1) under H1
            nonc = alpha_m * np.sqrt(T) / sigma
            t_power = 1.0 - (normal_cdf(1.96 - nonc) - normal_cdf(-1.96 - nonc))
            rows.append({
                "alpha_annual": alpha_ann * 100,
                "horizon_months": T,
                "reject_anytime": rej_any / B,
                "reject_final": rej_final / B,
                "t_test_power": t_power,
                "n_sim": B,
            })
    return pd.DataFrame(rows)


def normal_cdf(x: float) -> float:
    import math
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
