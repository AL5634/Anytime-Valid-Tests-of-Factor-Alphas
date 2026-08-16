"""
oos_horse_race.py — Out-of-sample horse race between factor screening methods.

We compare four factor-discovery criteria on a common train/test split:

    1. E-value screening          : e-value >= threshold (anytime-valid)
    2. HLZ (2016)                 : |t_alpha| > 3.0 (multiple-testing corrected)
    3. Empirical-Bayes shrinkage  : t-ratio shrunk toward zero, keeping
                                    factors whose shrunk |t| exceeds a
                                    q-quantile cutoff (a lightweight
                                    shrinkage screen in the spirit of
                                    Jensen, Kelly & Pedersen 2023, not the
                                    full hierarchical-Bayes model)
    4. Traditional                : |t_alpha| > 2.0

For fair comparison we select an equal number of factors per method
(ranked by strength within each method) and evaluate each portfolio on
fully out-of-sample data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..evalue.batch import compute_all_factor_evals


def _screen_regression_t(data, ff, factor_model, t_cut):
    """Screen factors whose OLS alpha t-stat exceeds t_cut."""
    facs = list(ff.columns) if factor_model.lower() != "capm" \
        else [c for c in ff.columns if c == "MKT"]
    selected = []
    ts = {}
    for f in data.columns:
        y = data[f].dropna()
        common = y.index.intersection(ff.index)
        if len(common) < 30:
            continue
        y = y.loc[common]
        X = sm.add_constant(ff.loc[common, facs])
        try:
            res = sm.OLS(y, X).fit()
            t = res.tvalues.iloc[0]
        except Exception:
            continue
        ts[f] = t
        if abs(t) > t_cut:
            selected.append(f)
    return selected, ts


def screen_evalue(evalue_results: pd.DataFrame, threshold: float) -> list:
    """Screen using final e-value threshold."""
    return evalue_results[evalue_results["evalue_final"] >= threshold].index.tolist()


def screen_hlz(data, ff, factor_model, t_cut=3.0):
    return _screen_regression_t(data, ff, factor_model, t_cut)[0]


def screen_traditional(data, ff, factor_model, t_cut=2.0):
    return _screen_regression_t(data, ff, factor_model, t_cut)[0]


def screen_empirical_bayes(data, ff, factor_model, q=0.05):
    """
    Empirical-Bayes shrinkage screen in the spirit of, but not identical to,
    the hierarchical-Bayesian factor selection of Jensen, Kelly & Pedersen
    (2023). OLS t-statistics are shrunk toward zero with an
    empirical-Bayes variance and factors are kept when their shrunk |t|
    exceeds a q-quantile cutoff. Unlike the full JKP model, this version
    does not share information across thematic clusters or countries.
    """
    _, ts = _screen_regression_t(data, ff, factor_model, t_cut=0.0)
    # Precision-weighted shrinkage mapping t -> effective |t|
    ts = pd.Series(ts).dropna()
    # approximate empirical-Bayes shrinkage
    tau2 = max(ts.var() - 1.0, 1e-8)
    shrink = tau2 / (tau2 + 1.0)
    eff_t = np.sqrt(shrink) * ts
    # keep those whose shrunk |t| ~ exceeds q-based threshold
    thr = abs(np.quantile(eff_t[eff_t != 0], 1 - q)) if (eff_t != 0).any() else 2.0
    return eff_t[abs(eff_t) > max(thr, 1.64)].index.tolist(), eff_t


def _top_n(selected, scores, n):
    """Retain the top-N factors of `selected` ranked by the method's OWN
    strength criterion (`scores`, a {factor: strength} mapping). Ranking
    every method by a common generic measure would make the count-equalized
    comparison vacuous (identical portfolios), so each method ranks by its
    own score: e-value for the E-value screen, |t| for the t screens, and
    shrunk |t| for the empirical-Bayes screen."""
    if n is None:
        return selected
    ranked = sorted(selected, key=lambda f: abs(scores.get(f, 0.0)),
                    reverse=True)
    return ranked[:n]


def compute_oos_performance(factors, factor_data, ff, oos_start, oos_end,
                            factor_model="CAPM"):
    """Equal-weight long-short portfolio of selected factors, OOS metrics."""
    y = factor_data.loc[oos_start:oos_end, factors]
    port = y.mean(axis=1).dropna()

    facs = ["MKT"] if factor_model.lower() == "capm" else list(ff.columns)
    common = port.index.intersection(ff.index)
    X = sm.add_constant(ff.loc[common, facs])
    res = sm.OLS(port.loc[common], X).fit()

    alpha_m = res.params.iloc[0]
    t_alpha = res.tvalues.iloc[0]
    ann_alpha = alpha_m * 12

    sharpe = port.mean() / port.std(ddof=1) * np.sqrt(12) if port.std() > 0 else np.nan

    cum = (1 + port).cumprod()
    # store a convenience field for figure assembly
    port_obj = port
    dd = (cum - cum.cummax()) / cum.cummax()
    max_dd = dd.min()

    # alpha decay: first half vs second half of OOS
    mid = port.index[len(port) // 2]
    def _alpha(seg):
        Xs = sm.add_constant(ff.loc[seg.index.intersection(ff.index), facs])
        rs = sm.OLS(port.loc[seg.index.intersection(Xs.index)], Xs).fit()
        return rs.params.iloc[0] * 12
    try:
        half1 = _alpha(port[port.index <= mid])
        half2 = _alpha(port[port.index > mid])
        decay = (half2 - half1)
    except Exception:
        decay = np.nan

    return {
        "n_factors": len(factors),
        "alpha_monthly_bps": alpha_m * 1e4,
        "alpha_annualized_pct": ann_alpha * 100,
        "t_alpha": t_alpha,
        "sharpe_annualized": sharpe,
        "max_drawdown": max_dd,
        "alpha_decay_half2_minus_half1_bps": decay * 1e4,
        "cum_returns": cum,
        "_portfolio": port_obj,
    }


def run_oos_horse_race(factor_data, ff, evalue_results, config,
                       selection_out=None, include_ebh=False):
    emp = config["empirical"]
    scr = config["screening"]
    fm = emp["factor_model"]

    train_start, train_end = emp["train_start"], emp["train_end"]
    test_start, test_end = emp["test_start"], emp["test_end"]

    train_data = factor_data.loc[train_start:train_end]
    train_ff = ff.loc[train_start:train_end]

    # 1. E-value screen -- compute the e-process on the TRAIN window only
    #    (strict OOS: the test window never enters the screen, fixing the
    #    full-sample leakage). evalue_results is kept as an optional
    #    fallback / cross-check, not used for the screen itself.
    train_ev = compute_all_factor_evals(
        train_data, train_ff, n_jobs=1,
        min_obs=scr.get("evalue_min_obs", 60),
        alpha=scr.get("evalue_alpha", 0.05))
    ev_sel = screen_evalue(train_ev, scr["evalue_threshold"])

    # 2-4. Regression-t based screens on the train window
    _, ts_all = _screen_regression_t(train_data, train_ff, fm, 0.0)
    hlz_sel = [f for f, t in ts_all.items() if abs(t) > scr["hlz_t_threshold"]]
    trad_sel = [f for f, t in ts_all.items() if abs(t) > scr["traditional_t_threshold"]]
    eb_sel, eb_eff_t = screen_empirical_bayes(train_data, train_ff, fm,
                                              scr["eb_fdr_q"])

    # Equalize counts for a fair comparison (each method ranks by its own
    # criterion: train e-value for E-value, |t| for the t screens, shrunk
    # |t| for the empirical-Bayes screen).
    n = scr.get("n_factors_equalize")
    if n is not None:
        ev_scores = train_ev["evalue_final"]
        ts_series = pd.Series(ts_all)
        ev_sel = _top_n(ev_sel, ev_scores, n)
        hlz_sel = _top_n(hlz_sel, ts_series, n)
        trad_sel = _top_n(trad_sel, ts_series, n)
        eb_sel = _top_n(eb_sel, eb_eff_t, n)

    if selection_out is not None:
        import json
        selection_out = Path(selection_out)
        selection_out.parent.mkdir(parents=True, exist_ok=True)
        with open(selection_out, "w") as f:
            json.dump({"ev": ev_sel, "hlz": hlz_sel,
                       "trad": trad_sel, "eb": eb_sel}, f)

    # e-BH (Wang & Ramdas 2022): zoo-wide FDR control on the train-window
    # terminal e-values. Reported as a fifth screening rule in the natural
    # fork (main comparison only; the count-equalized runs keep four rules).
    if include_ebh and n is None:
        from ..evalue.ebh import ebh_rejection_set
        alpha = scr.get("evalue_alpha", 0.05)
        ebh_sel = sorted(ebh_rejection_set(train_ev["evalue_final"], alpha=alpha))

    rows = []
    cum_paths = {}
    for method, sel in [("E-value", ev_sel), ("HLZ t>3.0", hlz_sel),
                        ("EB-shrink", eb_sel), ("t>2.0", trad_sel)]:
        if not sel:
            continue
        perf = compute_oos_performance(sel, factor_data, ff,
                                       test_start, test_end, fm)
        perf["method"] = method
        rows.append(perf)
        cum_paths[method] = perf.pop("_portfolio")

    if include_ebh and n is None and ebh_sel:
        perf = compute_oos_performance(ebh_sel, factor_data, ff,
                                       test_start, test_end, fm)
        perf["method"] = "E-value (e-BH)"
        rows.append(perf)
        cum_paths[perf["method"]] = perf.pop("_portfolio")

    cols = ["method", "n_factors", "alpha_monthly_bps", "alpha_annualized_pct",
            "t_alpha", "sharpe_annualized", "max_drawdown",
            "alpha_decay_half2_minus_half1_bps"]
    return pd.DataFrame(rows)[cols], cum_paths
