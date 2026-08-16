"""
internet_appendix.py — Internet-appendix diagnostics and robustness tables.

Outputs (data/results/):
  residual_tails.csv      — FF5 no-intercept residual skewness / excess
                            kurtosis, split by rejection status.
  oos_factor_corr.csv     — average / max pairwise correlation of the OOS
                            selected sets (threshold-20 and e-BH).
  oos_cat_equalweight.csv — category-equal-weighted OOS alpha (robustness).
  first_rejection_timing.csv — first-rejection vs publication-year timing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

FF_NAMES = ["mkt", "smb", "hml", "rmw", "cma"]


def _ff5_alpha(port: pd.Series, ff: pd.DataFrame) -> tuple[float, float]:
    X = sm.add_constant(ff.reindex(port.index)).dropna()
    y = port.reindex(X.index)
    r = sm.OLS(y, X).fit()
    return float(r.params.iloc[0] * 1200), float(r.tvalues.iloc[0])


def residual_tails(base: Path) -> None:
    univ = pd.read_csv(base / "data/processed/factor_universe.csv",
                       index_col=0, parse_dates=True)
    ev = pd.read_csv(base / "data/results/evalue_results.csv", index_col=0)
    ff = univ[[c for c in FF_NAMES]]
    zoo = [c for c in univ.columns if c not in FF_NAMES]

    rows = []
    for f in zoo:
        y = univ[f].dropna()
        X = sm.add_constant(ff.reindex(y.index)).dropna()
        y = y.reindex(X.index)
        res = sm.OLS(y, X).fit().resid
        rows.append({
            "factor": f,
            "skew": float(pd.Series(res).skew()),
            "exkurt": float(pd.Series(res).kurt()),
            "rejected": bool(ev.loc[f, "evalue_final"] >= 20),
        })
    df = pd.DataFrame(rows)
    df.to_csv(base / "data/results/residual_tails.csv", index=False)

    g = df.groupby("rejected")[["skew", "exkurt"]].median()
    print("residual tails (median skew / ex-kurtosis):")
    print(g.to_string())


def oos_corr(base: Path) -> None:
    univ = pd.read_csv(base / "data/processed/factor_universe.csv",
                       index_col=0, parse_dates=True)
    ev = pd.read_csv(base / "data/results/evalue_results.csv", index_col=0)
    with open(base / "data/results/oos_selections.json") as f:
        sel = json.load(f)
    oos = univ.loc["2006-01":"2023-12"]
    rows = []
    for label, factors in [("E-value (threshold 20)", sel["ev"]),
                           ("e-BH (full sample)", None)]:
        if factors is None:
            factors = list(ev.index[ev["ebh_reject"]])
        factors = [f for f in factors if f in oos.columns]
        if len(factors) < 2:
            continue
        corr = oos[factors].corr().to_numpy()
        n = len(factors)
        off = corr[~np.eye(n, dtype=bool)]
        rows.append({
            "selection": label,
            "n_factors": n,
            "mean_corr": float(off.mean()),
            "max_corr": float(off.max()),
            "share_above_0.5": float((off > 0.5).mean()),
        })
    pd.DataFrame(rows).to_csv(base / "data/results/oos_factor_corr.csv",
                              index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def oos_cat_equalweight(base: Path) -> None:
    univ = pd.read_csv(base / "data/processed/factor_universe.csv",
                       index_col=0, parse_dates=True)
    ev = pd.read_csv(base / "data/results/evalue_results.csv", index_col=0)
    ff = univ[[c for c in FF_NAMES]]
    with open(base / "data/results/oos_selections.json") as f:
        sel = json.load(f)
    oos = univ.loc["2006-01":"2023-12"]

    factors = [f for f in sel["ev"] if f in oos.columns]
    cats = ev.loc[factors, "cat_economic"]
    # equal weight within category, then equal weight across categories
    cat_rets = pd.DataFrame({c: oos[factors].loc[:, (cats == c).to_numpy()].mean(axis=1)
                             for c in cats.unique()})
    port = cat_rets.mean(axis=1)
    a, t = _ff5_alpha(port, ff)
    print(f"category-equal-weight OOS alpha: {a:.2f}%/yr (t={t:.2f})")
    pd.DataFrame([{"scheme": "category-equal-weight",
                   "alpha_annualized_pct": a, "t_alpha": t}]).to_csv(
        base / "data/results/oos_cat_equalweight.csv", index=False)


def first_rejection_timing(base: Path) -> None:
    ev = pd.read_csv(base / "data/results/evalue_results.csv", index_col=0)
    rej = ev[ev["evalue_final"] >= 20].copy()
    rej["first_rej_year"] = pd.to_numeric(
        rej["first_rejection_date"].str[:4], errors="coerce")
    rej["pub_year"] = pd.to_numeric(rej["pub_year"], errors="coerce")
    rej["pre_pub"] = rej["first_rej_year"] < rej["pub_year"]
    rej["lead_years"] = rej["pub_year"] - rej["first_rej_year"]

    n_rej = len(rej)
    n_pre = int(rej["pre_pub"].sum())
    n_pre_decay = int((rej["pre_pub"] & (rej["post_pub_decay_ratio"] < 1)).sum())
    n_lead10 = int((rej["lead_years"] >= 10).sum())

    out = pd.DataFrame([{
        "n_rejectors": n_rej,
        "n_reject_before_pub": n_pre,
        "n_reject_before_pub_and_decay": n_pre_decay,
        "n_reject_at_least_10y_before_pub": n_lead10,
    }])
    out.to_csv(base / "data/results/first_rejection_timing.csv", index=False)
    print(out.to_string(index=False))


def main() -> None:
    base = Path(__file__).resolve().parents[2]
    residual_tails(base)
    oos_corr(base)
    oos_cat_equalweight(base)
    first_rejection_timing(base)


if __name__ == "__main__":
    main()
