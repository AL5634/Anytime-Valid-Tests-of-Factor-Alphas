"""
oos_differentiation.py — Out-of-sample differentiation of the screening rules.

B-1 (exclusion value): the e-value screen (62 train-window predictors) is
almost a strict subset of the |t|>2 screen (157): 60 of its 62 selections are
also in |t|>2, and |t|>2 adds 97 predictors the e-value rejects. We compute
the OOS (2006:01-2023:12) FF5 alpha and Sharpe of the core set (ev intersect
trad), the excluded set (trad minus ev), and the e-BH counterparts, to ask
whether what e-value excludes is economically weak out of sample.

B-2 (late sub-window): with the train selections frozen (2005:12), we evaluate
every rule's portfolio on the early (2006:01-2014:12) and the late
(2015:01-2023:12) sub-windows to see whether the e-value-selected factors
persist most recently.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.oos_horse_race import compute_oos_performance
from src.evalue.batch import compute_all_factor_evals
from src.evalue.ebh import ebh_rejection_set

FF_NAMES = ["mkt", "smb", "hml", "rmw", "cma"]
TRAIN_START, TRAIN_END = "1963-07", "2005-12"
OOS_START, OOS_END = "2006-01", "2023-12"
EARLY_END = "2014-12"
LATE_START = "2015-01"


def _load():
    base = Path(__file__).resolve().parents[2]
    univ = pd.read_csv(base / "data/processed/factor_universe.csv",
                       index_col=0, parse_dates=True)
    ff = univ[[c for c in FF_NAMES]]
    zoo = [c for c in univ.columns if c not in FF_NAMES]
    with open(base / "data/results/oos_selections.json") as f:
        sel = json.load(f)
    ev, hlz, trad, eb = (set(sel["ev"]), set(sel["hlz"]),
                         set(sel["trad"]), set(sel["eb"]))
    # train-window e-BH selection (not persisted in oos_selections.json)
    train_ev = compute_all_factor_evals(univ[zoo].loc[TRAIN_START:TRAIN_END],
                                        ff.loc[TRAIN_START:TRAIN_END],
                                        n_jobs=-1, min_obs=60, alpha=0.05)
    ebh = set(ebh_rejection_set(train_ev["evalue_final"], alpha=0.05))
    return univ, ff, {"ev": ev, "hlz": hlz, "trad": trad, "eb": eb, "ebh": ebh}


def _perf(factors, univ, ff, start, end):
    factors = [f for f in factors if f in univ.columns]
    if not factors:
        return {"n_factors": 0, "alpha_annualized_pct": float("nan"),
                "t_alpha": float("nan"), "sharpe_annualized": float("nan")}
    r = compute_oos_performance(factors, univ, ff, start, end, "FF5")
    return {k: r[k] for k in ("n_factors", "alpha_annualized_pct",
                              "t_alpha", "sharpe_annualized")}


def b1_exclusion(univ, ff, sel):
    ev, trad, ebh = sel["ev"], sel["trad"], sel["ebh"]
    rows = []
    port = {
        "E-value (full, 62)": ev,
        "Core = E-value $\\cap$ |t|>2": ev & trad,
        "Excluded by E-value (|t|>2 - E-value)": trad - ev,
        "E-value only": ev - trad,
        "|t|>2 (full, 157)": trad,
        "E-value e-BH (43)": ebh,
        "Excluded by e-BH (|t|>2 - e-BH)": trad - ebh,
    }
    for label, facs in port.items():
        p = _perf(list(facs), univ, ff, OOS_START, OOS_END)
        p["portfolio"] = label
        rows.append(p)
    return pd.DataFrame(rows)[["portfolio", "n_factors", "alpha_annualized_pct",
                               "t_alpha", "sharpe_annualized"]]


def b2_late_holdout(univ, ff, sel):
    rows = []
    for rule, facs in sel.items():
        if not facs:
            continue
        for label, s, e in [("Early (2006-2014)", OOS_START, EARLY_END),
                            ("Late (2015-2023)", LATE_START, OOS_END)]:
            p = _perf(list(facs), univ, ff, s, e)
            p["rule"] = rule
            p["window"] = label
            rows.append(p)
    return pd.DataFrame(rows)[["rule", "window", "n_factors",
                               "alpha_annualized_pct", "t_alpha",
                               "sharpe_annualized"]]


def main():
    base = Path(__file__).resolve().parents[2]
    univ, ff, sel = _load()
    e = b1_exclusion(univ, ff, sel)
    e.to_csv(base / "data/results/oos_exclusion.csv", index=False)
    print("=== B-1 exclusion value (OOS 2006-2023) ===")
    print(e.to_string(index=False))
    h = b2_late_holdout(univ, ff, sel)
    h.to_csv(base / "data/results/oos_lateholdout.csv", index=False)
    print("\n=== B-2 late sub-window ===")
    print(h.to_string(index=False))


if __name__ == "__main__":
    main()
