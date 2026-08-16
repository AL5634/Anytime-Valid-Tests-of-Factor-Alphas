"""
skew_size.py — Monte-Carlo size study of the regression-alpha e-process under
skewed and heavy-tailed residual DGPs.

The anytime-valid guarantee rests on a maintained condition: the clipped
increment Z_t has zero conditional mean, which holds when residuals are
conditionally symmetric about zero (or truncation is negligible). Because
anomaly residuals are skew and heavy-tailed (median skew -0.18, median
excess kurtosis 4.77 among rejectors, with far more extreme values in the
tails), we check whether the empirical size stays at or below the nominal
5% under such shapes.

Experiment A (parametric, iid): r_t = beta' F_t + eps_t with alpha = 0 and
eps drawn from Normal / skew-normal / Student-t / two-piece t calibrated to
the empirical residual moments. No serial dependence, so it isolates the
effect of skewness and heavy tails on size.

Experiment B (empirical residuals): for each of the 212 CZ predictors we take
the full-sample FF5 no-intercept residuals, demean them (forcing alpha = 0,
the null), and run the e-process on the demeaned residual series with the FF5
factors. This preserves the real distributional shape AND any serial
dependence. The fraction of the 212 paths that ever cross 1/alpha is the
empirical size under realistic residual shapes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evalue.core import regression_alpha_evalue

FF_NAMES = ["mkt", "smb", "hml", "rmw", "cma"]
ALPHA = 0.05
THRESHOLD = 1.0 / ALPHA
CLIP = 3.0
B_REPS = 2000
HORIZONS = (240, 726)


def _skew_normal_delta(skew: float) -> float:
    """Delta parametrisation for a skew-normal with target skewness (sign
    preserved). Uses the standard formula
    skew = (4-pi)/2 * (d sqrt(2/pi))^3 / (1 - 2 d^2/pi)^{3/2}, d in [0,1)."""
    if abs(skew) < 1e-9:
        return 0.0
    sign = 1.0 if skew > 0 else -1.0
    s = abs(skew)

    def sk(d):
        a = d ** 2
        dd = np.sqrt(2.0 / np.pi) * d
        return (4.0 - np.pi) / 2.0 * dd ** 3 / (1.0 - 2.0 * a / np.pi) ** 1.5

    lo, hi = 0.0, 0.9999
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if sk(mid) < s:
            lo = mid
        else:
            hi = mid
    return sign * 0.5 * (lo + hi)


def _skewed_normal(rng, skew, size):
    d = _skew_normal_delta(skew)
    if abs(d) < 1e-12:
        return rng.normal(size=size)
    z0 = rng.normal(size=size)
    z1 = rng.normal(size=size)
    return d * np.abs(z0) + np.sqrt(1.0 - d ** 2) * z1


def _two_piece_t(rng, df, skew_param, size):
    """Two-piece (split) Student-t: negative half scaled by (1+skew_param),
    positive half by (1-skew_param), then centred and standardised. Yields
    arbitrary negative skew with heavy tails."""
    u = rng.standard_t(df, size=size)
    x = u * np.where(u < 0, 1.0 + skew_param, 1.0 - skew_param)
    return (x - x.mean()) / x.std()


def _resid_dgp_components():
    base = Path(__file__).resolve().parents[2]
    panel = pd.read_csv(base / "data/processed/factor_universe.csv",
                        index_col=0, parse_dates=True)
    ff = panel[[c for c in FF_NAMES]]
    import statsmodels.api as sm
    zoo = [c for c in panel.columns if c not in FF_NAMES]
    sigmas = {}
    for f in zoo:
        y = panel[f].dropna()
        X = sm.add_constant(ff.reindex(y.index)).dropna()
        y = y.reindex(X.index)
        if len(y) < 120:
            continue
        sigmas[f] = float(np.sqrt(sm.OLS(y, X).fit().mse_resid))
    med = pd.Series(sigmas).sort_values()
    f_star = med.index[len(med) // 2]
    y = panel[f_star].dropna()
    X = sm.add_constant(ff.reindex(y.index)).dropna()
    y = y.reindex(X.index)
    res = sm.OLS(y, X).fit()
    return np.asarray(res.params[1:]), float(np.sqrt(res.mse_resid)), ff


def _one_run(r_series, ff_sub, min_obs=60):
    e = regression_alpha_evalue(r_series, ff_sub, min_obs=min_obs,
                                clip_scale=CLIP, alpha=ALPHA)["e_path"]
    return bool((e >= THRESHOLD).any())


def _standardize(z):
    return (z - np.mean(z)) / np.std(z)


def experiment_a():
    beta, sigma, ff = _resid_dgp_components()
    rows = []
    # (label, kind, params) — two-piece t labels carry achieved skew/kurtosis
    dgps = [
        ("Normal", "normal", {}),
        ("Skew-normal, skew=-0.18", "skewn", {"skew": -0.18}),
        ("Skew-normal, skew=-1.0", "skewn", {"skew": -1.0}),
        ("Student-t, df=5", "t", {"df": 5}),
        ("Two-piece t, df=5, skew~-1.3", "twopiece", {"df": 5, "sp": 0.3}),
        ("Two-piece t, df=4, skew~-2.8", "twopiece", {"df": 4, "sp": 0.6}),
        ("Two-piece t, df=3, skew~-5.7", "twopiece", {"df": 3, "sp": 0.6}),
    ]
    for T in HORIZONS:
        ff_window = ff.iloc[-T:].to_numpy()
        ff_sub = ff.iloc[-T:]
        idx = ff_sub.index
        for label, kind, kw in dgps:
            rng = np.random.default_rng(seed=sum(map(ord, label)) + T)
            rej = 0
            bind_sum = 0.0
            bias_sum = 0.0
            for _ in range(B_REPS):
                if kind == "normal":
                    eps = rng.normal(0.0, sigma, T)
                elif kind == "skewn":
                    eps = sigma * _standardize(_skewed_normal(rng, kw["skew"], T))
                elif kind == "t":
                    eps = sigma * (rng.standard_t(kw["df"], size=T)
                                   / np.sqrt(kw["df"] / (kw["df"] - 2)))
                elif kind == "twopiece":
                    eps = sigma * _two_piece_t(rng, kw["df"], kw["sp"], T)
                r = np.asarray(ff_window) @ beta + eps
                r_series = pd.Series(r, index=idx)
                if _one_run(r_series, ff_sub):
                    rej += 1
                B = CLIP * sigma
                clipped = np.clip(eps / B, -1.0, 1.0)
                bind_sum += float((np.abs(eps / B) > 1.0).mean())
                bias_sum += float(clipped.mean())
            rows.append({
                "experiment": "parametric",
                "dgp": label,
                "T": T,
                "n_sim": B_REPS,
                "empirical_size": rej / B_REPS,
                "clip_bind_freq": bind_sum / B_REPS,
                "truncation_bias": bias_sum / B_REPS,
            })
    return pd.DataFrame(rows)


def experiment_b():
    base = Path(__file__).resolve().parents[2]
    panel = pd.read_csv(base / "data/processed/factor_universe.csv",
                        index_col=0, parse_dates=True)
    ff = panel[[c for c in FF_NAMES]]
    zoo = [c for c in panel.columns if c not in FF_NAMES]

    import statsmodels.api as sm
    n_rej = 0
    n_factors = 0
    for f in zoo:
        y = panel[f].dropna()
        X = sm.add_constant(ff.reindex(y.index)).dropna()
        y = y.reindex(X.index)
        if len(y) < 200:
            continue
        res = sm.OLS(y, X).fit().resid
        demeaned = (res - res.mean()).rename("r")
        common = demeaned.index.intersection(ff.index)
        if _one_run(demeaned.loc[common], ff.loc[common]):
            n_rej += 1
        n_factors += 1
    return pd.DataFrame([{
        "experiment": "empirical",
        "dgp": "demeaned FF5 residuals (212 predictors)",
        "T": int(len(ff)),
        "n_sim": n_factors,
        "empirical_size": n_rej / n_factors,
        "clip_bind_freq": np.nan,
        "truncation_bias": np.nan,
    }])


def main():
    base = Path(__file__).resolve().parents[2]
    out = base / "data/results/skew_size.csv"
    a = experiment_a()
    b = experiment_b()
    df = pd.concat([a, b], ignore_index=True)
    df.to_csv(out, index=False)
    print(df[["experiment", "dgp", "T", "empirical_size", "clip_bind_freq",
              "truncation_bias"]].to_string(index=False))


if __name__ == "__main__":
    main()
