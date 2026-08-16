"""
verify_oos_independent.py — Independent, from-scratch verification of the
out-of-sample screening results (Table 4 of the manuscript).

The main pipeline's OOS numbers (oos_comparison.csv / oos_selections.json)
are regenerated here WITHOUT importing src.analysis.oos_horse_race or
src.evalue.cross_market:

  1. panel : built fresh from PredictorLSretWide.csv (percents -> decimals,
             month-end index) joined to the frozen FF5 panel
             (data/downloaded/ff5_panel.csv); window 1963:07-2023:12,
             factors with >= 60 non-missing months.
  2. train e-values : the paper's bounded-betting e-process re-implemented
             from the formulas (expanding no-intercept projection, clip
             bound B_t = 3 sigma_{t-1}, lambda_t = 2 * mean(Z), cap .99).
  3. train t-stats  : plain OLS alpha t-statistics on the train window.
  4. screens        : E-value >= 20; e-BH (Wang-Ramdas) on train e-values;
             |t|>3; |t|>2; empirical-Bayes shrinkage.
  5. portfolios     : equal-weight long-short of each screen's factors over
             the test window; FF5 regression with intercept; alpha (ann. %),
             t, Sharpe, MaxDD computed with plain pandas/statsmodels.

Each row of oos_comparison.csv and each list of oos_selections.json is
then compared with the independently derived numbers.

Run: python3 scripts/verify_oos_independent.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

MIN_OBS = 60
TRAIN_START, TRAIN_END = "1963-07", "2005-12"
TEST_START, TEST_END = "2006-01", "2023-12"
FF5 = ["mkt", "smb", "hml", "rmw", "cma"]


# ---------------------------------------------------------------- panel
def build_panel() -> pd.DataFrame:
    raw = pd.read_csv(BASE / "PredictorLSretWide.csv")
    date = pd.to_datetime(raw["date"], format="%Y-%m-%d", errors="coerce")
    panel = raw.drop(columns=["date"]).apply(pd.to_numeric, errors="coerce")
    panel.index = date + pd.offsets.MonthEnd(0)
    panel = panel.sort_index() / 100.0
    panel = panel[panel.index <= pd.Timestamp("2023-12-31")]
    panel = panel[[c for c in panel.columns if panel[c].notna().sum() >= MIN_OBS]]

    ff = pd.read_csv(BASE / "data/downloaded/ff5_panel.csv", comment="#",
                     index_col=0, parse_dates=True)
    ff.columns = [c.lower() for c in ff.columns]
    ff.index = ff.index + pd.offsets.MonthEnd(0)
    ff = {k: ff[k].dropna() for k in FF5}
    ff = pd.DataFrame(ff)

    u = ff.join(panel, how="inner").loc[TRAIN_START:TEST_END]
    u = u[[c for c in u.columns if u[c].notna().sum() >= MIN_OBS]]
    return u


# ---------------------------------------------------------------- e-process
def evalue_path(y: np.ndarray, ff: np.ndarray, min_obs: int = MIN_OBS,
                clip_scale: float = 3.0, burn: int = 24) -> np.ndarray:
    n = len(y)
    resid = np.full(n, np.nan)
    for t in range(min_obs, n):
        fw, yw = ff[:t], y[:t]
        beta = np.linalg.pinv(fw.T @ fw + 1e-10 * np.eye(fw.shape[1])) @ (fw.T @ yw)
        resid[t] = y[t] - ff[t] @ beta
    start = min_obs + burn
    e = 1.0
    ep = np.zeros(n - start)
    run = 0.0
    for j, t in enumerate(range(start, n)):
        sig = float(np.nanstd(resid[min_obs:t], ddof=1) or 1.0)
        B = clip_scale * sig
        z = float(np.clip(resid[t] / B, -1.0, 1.0))
        lam = 0.0 if j == 0 else np.clip((run / j) * 2.0, -0.99, 0.99)
        e = e * (1.0 + lam * z)
        ep[j] = e
        run += z
    return ep


def ebh_set(evalues: np.ndarray) -> set:
    e = np.sort(evalues[np.isfinite(evalues)])[::-1]
    K = len(e)
    k_hat = 0
    for k in range(1, K + 1):
        if e[k - 1] >= K / (0.05 * k):
            k_hat = k
    thr = K / (0.05 * k_hat)
    return set(np.flatnonzero(evalues >= thr).tolist())


# ---------------------------------------------------------------- screens
def ols_t(y: pd.Series, ff: pd.DataFrame) -> float:
    y = y.dropna()
    common = y.index.intersection(ff.index)
    X = sm.add_constant(ff.loc[common])
    return float(sm.OLS(y.loc[common], X).fit().tvalues.iloc[0])


def eb_shrink_selection(ts: pd.Series, q: float = 0.05) -> list:
    tau2 = max(ts.var() - 1.0, 1e-8)
    shrink = tau2 / (tau2 + 1.0)
    eff_t = np.sqrt(shrink) * ts
    thr = abs(np.quantile(eff_t[eff_t != 0], 1 - q)) if (eff_t != 0).any() else 2.0
    return eff_t[abs(eff_t) > max(thr, 1.64)].index.tolist()


# ---------------------------------------------------------------- portfolio
def oos_perf(factors, u: pd.DataFrame, ff: pd.DataFrame) -> dict:
    port = u.loc[TEST_START:TEST_END, factors].mean(axis=1).dropna()
    common = port.index.intersection(ff.index)
    X = sm.add_constant(ff.loc[common])
    res = sm.OLS(port.loc[common], X).fit()
    alpha_m = float(res.params.iloc[0])
    ann = alpha_m * 12 * 100
    t = float(res.tvalues.iloc[0])
    sharpe = float(port.mean() / port.std(ddof=1) * np.sqrt(12))
    cum = (1 + port).cumprod()
    mdd = float(((cum - cum.cummax()) / cum.cummax()).min())
    return {"n_factors": len(factors), "alpha_annualized_pct": ann,
            "t_alpha": t, "sharpe_annualized": sharpe, "max_drawdown": mdd}


def main() -> int:
    print("building panel from raw inputs (independent)...")
    u = build_panel()
    zoo = [c for c in u.columns if c not in FF5]
    ff = u[FF5]
    train = u.loc[TRAIN_START:TRAIN_END]
    train_ff = ff.reindex(train.index)

    print("train e-values (from-scratch)...")
    ev_final = {}
    for f in zoo:
        al = pd.concat([train[f], train_ff], axis=1).dropna()
        if len(al) < MIN_OBS + 24 + 1:
            ev_final[f] = np.nan
            continue
        ep = evalue_path(al.iloc[:, 0].to_numpy(), al.iloc[:, 1:].to_numpy())
        ev_final[f] = float(ep[-1]) if len(ep) else np.nan
    ev = pd.Series(ev_final)
    ev_sel = sorted(ev[ev >= 20.0].index)
    ebh_idx = ebh_set(ev.to_numpy())
    ebh_sel = sorted(ev.index[np.asarray(sorted(ebh_idx))].tolist())

    print("train t-stats (plain OLS)...")
    ts = {f: ols_t(train[f], ff) for f in zoo}
    ts = pd.Series(ts).dropna()
    hlz_sel = sorted(ts[abs(ts) > 3.0].index)
    trad_sel = sorted(ts[abs(ts) > 2.0].index)
    eb_sel = sorted(eb_shrink_selection(ts))

    screens = {"E-value": ev_sel, "HLZ t>3.0": hlz_sel,
               "EB-shrink": eb_sel, "t>2.0": trad_sel,
               "E-value (e-BH)": ebh_sel}

    print("OOS portfolios (independent)...")
    indep = {m: oos_perf(sel, u, ff) for m, sel in screens.items() if sel}

    # ---- compare selections ----
    sel_json = json.load(open(BASE / "data/results/oos_selections.json"))
    map_ = {"E-value": "ev", "HLZ t>3.0": "hlz", "t>2.0": "trad",
            "EB-shrink": "eb"}
    sel_ok = True
    for m, key in map_.items():
        a, b = set(screens[m]), set(sel_json[key])
        if a != b:
            sel_ok = False
            print(f"  SELECTION MISMATCH {m}: only-indep={sorted(a-b)[:4]} only-pipe={sorted(b-a)[:4]}")
    print(f"[{'PASS' if sel_ok else 'FAIL'}] selections (4 lists, independent vs pipeline)")

    # ---- compare performance ----
    df = pd.read_csv(BASE / "data/results/oos_comparison.csv")
    perf_ok = True
    for _, r in df.iterrows():
        m = r["method"]
        if m not in indep:
            perf_ok = False
            print(f"  MISSING {m}")
            continue
        x = indep[m]
        checks = [
            ("n_factors", r["n_factors"], x["n_factors"], 0),
            ("alpha_annualized_pct", r["alpha_annualized_pct"], x["alpha_annualized_pct"], 0.02),
            ("t_alpha", r["t_alpha"], x["t_alpha"], 0.05),
            ("sharpe_annualized", r["sharpe_annualized"], x["sharpe_annualized"], 0.02),
            ("max_drawdown", r["max_drawdown"], x["max_drawdown"], 0.01),
        ]
        for col, pipe, ind, tol in checks:
            if abs(pipe - ind) > tol:
                perf_ok = False
                print(f"  PERF MISMATCH {m}.{col}: pipeline={pipe:.4f} independent={ind:.4f}")
        if m == "E-value (e-BH)" and r["n_factors"] != 43:
            perf_ok = False
            print("  e-BH n != 43")

    print(f"[{'PASS' if perf_ok else 'FAIL'}] OOS performance (5 rules, independent vs oos_comparison.csv)")
    for m, x in indep.items():
        print(f"    {m:<16} N={x['n_factors']:<3} alpha={x['alpha_annualized_pct']:.2f} "
              f"t={x['t_alpha']:.2f} Sharpe={x['sharpe_annualized']:.2f} MaxDD={x['max_drawdown']:.3f}")
    return 0 if (sel_ok and perf_ok) else 1


if __name__ == "__main__":
    sys.exit(main())