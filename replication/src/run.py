"""
run.py — End-to-end runner for the E-value Factor Zoo analysis.

Usage:
    python -m src.run --stage all        # download data + run everything
    python -m src.run --stage data       # download / build factor panel
    python -m src.run --stage evalue     # compute e-values for all factors
    python -m src.run --stage oos        # run the out-of-sample horse race
    python -m src.run --stage figures    # render figures

Outputs:
    data/results/evalue_results.csv      # per-factor e-value diagnostics
    data/results/oos_comparison.csv      # OOS horse-race results
    output/figures/*.pdf                 # paper figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, paths
from src.data.download import load_factor_universe, get_ff_factor_frets
from src.data.factor_metadata import all_publication_dates
from src.data.cz import (load_cz_universe, load_cz_metadata,
                         cz_publication_dates)
from src.evalue.cross_market import compute_all_factor_evals
from src.analysis.oos_horse_race import run_oos_horse_race
from src.viz import figures


# Candidate "factor zoo" factors to test (the benchmark FF factors are the
# conditioning variables; the remaining long/short spreads are the zoo).
FF_NAMES = ["mkt", "smb", "hml", "rmw", "cma"]

# CZ factors are SignalDoc Acronyms (e.g. "Accruals"); the FF conditioning
# set is always the lowercase Ken French names regardless of universe.
OOS_SAMPLE_START = "1963-07"
OOS_SAMPLE_END = "2023-12"


def _ff_panel() -> pd.DataFrame:
    """Ken French five factors as a lowercase wide DataFrame, month-end index
    (the CZ panel uses month-end, so the FF index is normalised to match)."""
    ff = get_ff_factor_frets()
    ff = {k.lower(): ff[k.upper()].dropna() for k in ("MKT", "SMB", "HML", "RMW", "CMA")}
    out = pd.DataFrame(ff)
    out.index = out.index + pd.offsets.MonthEnd(0)
    return out


def load_data(config, p):
    universe = config["empirical"].get("universe", "kf")
    print(f"[1/4] Building factor universe ({universe}) ...")
    if universe == "cz":
        panel = load_cz_universe(min_obs=config["evalue"]["min_obs"])
        ff = _ff_panel()
        # align CZ predictors with the FF5 sample window (1963-07..2023-12)
        universe = ff.join(panel, how="inner")
        universe = universe.loc[OOS_SAMPLE_START:OOS_SAMPLE_END]
        universe = universe[[c for c in universe.columns
                             if universe[c].notna().sum()
                             >= config["evalue"]["min_obs"]]]
    else:
        universe = load_factor_universe(min_obs=config["evalue"]["min_obs"])
    p["processed"].mkdir(parents=True, exist_ok=True)
    universe.to_csv(p["processed"] / "factor_universe.csv")
    return universe


def _publication_dates(config, factor_data) -> dict[str, str]:
    """Publication-date map for the active universe."""
    if config["empirical"].get("universe", "kf") == "cz":
        meta = load_cz_metadata()
        dates = cz_publication_dates(meta)
        return {k: v for k, v in dates.items() if k in factor_data.columns}
    return all_publication_dates()


def stage_evalue(factor_data, config, p) -> pd.DataFrame:
    print("[2/4] Computing e-values for all candidate factors ...")
    fm = config["empirical"]["factor_model"].lower()
    ff_cols = config["empirical"][f"{fm}_factors"] if fm == "ff5" else (
        config["empirical"]["ff3_factors"] if fm == "ff3"
        else config["empirical"]["capm_factors"])
    # Data columns are lowercase (load_factor_universe); config lists them
    # in uppercase (e.g. MKT). Lowercase before selection so the FF factors
    # are actually used as the conditioning set (alpha adjustment).
    ff_cols = [c.lower() for c in ff_cols]
    ff_use = factor_data[[c for c in ff_cols if c in factor_data.columns]]
    zoo = [c for c in factor_data.columns if c not in FF_NAMES]

    res = compute_all_factor_evals(
        factor_data[zoo], ff_use,
        n_jobs=config["evalue"].get("n_jobs", -1),
        alpha=config["evalue"].get("alpha", 0.05),
        min_obs=config["evalue"].get("min_obs", 60),
        clip_scale=config["evalue"].get("clip_scale", 3.0),
        publication_dates=_publication_dates(config, factor_data))
    res = _attach_metadata(res, config)
    # zoo-wide FDR statement: e-BH (Wang & Ramdas 2022) on the terminal
    # e-values of the active universe.
    from src.evalue.ebh import ebh_rejection_set
    import json
    alpha = config["evalue"].get("alpha", 0.05)
    ebh_set = ebh_rejection_set(res["evalue_final"], alpha=alpha)
    res["ebh_reject"] = res.index.isin(ebh_set)
    with open(p["results"] / "ebh_summary.json", "w") as f:
        json.dump({"alpha": alpha, "n_tested": int(len(res)),
                   "n_rejected_threshold20": int((res["evalue_final"] >= 1.0 / alpha).sum()),
                   "n_rejected_ebh": int(res["ebh_reject"].sum()),
                   "rejected_ebh": sorted(res.index[res["ebh_reject"]].tolist())}, f)
    p["results"].mkdir(parents=True, exist_ok=True)
    res.to_csv(p["results"] / "evalue_results.csv")
    return res


def _attach_metadata(res: pd.DataFrame, config) -> pd.DataFrame:
    """Join SignalDoc metadata (category, sign, pub year) onto the results so
    tables/figures can summarise by economic category without extra plumbing."""
    if config["empirical"].get("universe", "kf") != "cz":
        return res
    meta = load_cz_metadata().set_index("acronym")
    cols = ["cat_economic", "cat_signal", "sign", "pub_year"]
    ann = meta[cols].reindex(res.index)
    return res.join(ann)


def stage_oos(factor_data, evalue_results, config, p):
    print("[3/4] OOS horse-race ...")
    fm = config["empirical"]["factor_model"].lower()
    ff_cols = config["empirical"][f"{fm}_factors"] if fm == "ff5" else (
        config["empirical"]["ff3_factors"] if fm == "ff3"
        else config["empirical"]["capm_factors"])
    ff_cols = [c.lower() for c in ff_cols]  # data columns are lowercase
    ff_use = factor_data[[c for c in ff_cols if c in factor_data.columns]]
    zoo = [c for c in factor_data.columns if c not in FF_NAMES]
    df, cum = run_oos_horse_race(factor_data[zoo], ff_use, evalue_results,
                                 config,
                                 selection_out=p["results"] / "oos_selections.json",
                                 include_ebh=True)
    df.to_csv(p["results"] / "oos_comparison.csv", index=False)
    # cache cumulative paths for Fig 7
    import pickle
    with open(p["results"] / "oos_cumulative.pkl", "wb") as f:
        pickle.dump(cum, f)

    # Equalized-N comparisons for the internet appendix (strict fairness:
    # every method selects exactly N top factors on the train window).
    import copy
    for n in (2, 5, 10):
        cfg = copy.deepcopy(config)
        cfg["screening"]["n_factors_equalize"] = n
        dfn, _ = run_oos_horse_race(factor_data[zoo], ff_use, evalue_results,
                                    cfg)
        dfn.to_csv(p["results"] / f"oos_comparison_equalized_n{n}.csv",
                   index=False)
    return df


def stage_figures(config, p, factor_data=None):
    print("[4/4] Figures ...")
    p["figures"].mkdir(parents=True, exist_ok=True)
    if factor_data is None and (p["processed"] / "factor_universe.csv").exists():
        factor_data = pd.read_csv(p["processed"] / "factor_universe.csv",
                                  index_col=0, parse_dates=True)
    if factor_data is not None:
        figures.render_real_figures(factor_data, p["results"], p["figures"],
                                    config)


def stage_robust(factor_data, config, p):
    """Power simulation + min_obs x clip sensitivity grid (active universe)."""
    import numpy as np
    from src.analysis.power import simulate_power
    from src.analysis.sensitivity import run_sensitivity_grid

    ff_use = factor_data[[c for c in FF_NAMES if c in factor_data.columns]]
    zoo = [c for c in factor_data.columns if c not in FF_NAMES]
    p["results"].mkdir(parents=True, exist_ok=True)

    print("[robust] Power simulation (B=800, T in {240,726}) ...")
    rng = np.random.default_rng(7)
    power = simulate_power(ff_use, rng)
    power.to_csv(p["results"] / "power_simulation.csv", index=False)

    print("[robust] Sensitivity grid (min_obs x clip_scale) ...")
    grid = run_sensitivity_grid(
        factor_data, ff_use, zoo,
        alpha=config["evalue"].get("alpha", 0.05),
        n_jobs=config["evalue"].get("n_jobs", -1))
    grid.to_csv(p["results"] / "sensitivity_grid.csv", index=False)


def stage_kf_comparison(config, p):
    """Regenerate the Ken French 16-factor universe as an appendix
    "public-subset" comparison for the internet appendix. Runs the full
    evalue + OOS pipeline into a SEPARATE results dir so the main CZ panel
    and results are not touched."""
    import copy
    from src.analysis.oos_horse_race import run_oos_horse_race

    cfg = copy.deepcopy(config)
    cfg["empirical"]["universe"] = "kf"
    kf_dir = Path("data/results_kf")
    kf_dir.mkdir(parents=True, exist_ok=True)

    print("[kf-cmp] Building Ken French 16-factor universe ...")
    univ = load_factor_universe(min_obs=config["evalue"]["min_obs"])
    # evaluate on the SAME calibrated window as the main CZ analysis so the
    # appendix comparison is apples-to-apples with the paper.
    univ = univ.loc[OOS_SAMPLE_START:OOS_SAMPLE_END]
    univ.to_csv(Path("data/processed/factor_universe_kf.csv"))

    ff_use = univ[[c for c in FF_NAMES if c in univ.columns]]
    zoo = [c for c in univ.columns if c not in FF_NAMES]

    print("[kf-cmp] E-values ...")
    from src.evalue.cross_market import compute_all_factor_evals
    res = compute_all_factor_evals(
        univ[zoo], ff_use,
        n_jobs=config["evalue"].get("n_jobs", -1),
        alpha=config["evalue"].get("alpha", 0.05),
        min_obs=config["evalue"].get("min_obs", 60),
        clip_scale=config["evalue"].get("clip_scale", 3.0),
        publication_dates=all_publication_dates())
    res.to_csv(kf_dir / "evalue_results.csv")

    print("[kf-cmp] OOS horse race ...")
    oos_cfg = copy.deepcopy(cfg)
    oos_cfg["screening"]["n_factors_equalize"] = None
    df, cum = run_oos_horse_race(univ[zoo], ff_use, res, oos_cfg,
                                 selection_out=kf_dir / "oos_selections.json")
    df.to_csv(kf_dir / "oos_comparison.csv", index=False)
    import pickle
    with open(kf_dir / "oos_cumulative.pkl", "wb") as f:
        pickle.dump(cum, f)
    for n in (2, 5, 10):
        cfg_n = copy.deepcopy(oos_cfg)
        cfg_n["screening"]["n_factors_equalize"] = n
        dfn, _ = run_oos_horse_race(univ[zoo], ff_use, res, cfg_n)
        dfn.to_csv(kf_dir / f"oos_comparison_equalized_n{n}.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "data", "evalue", "oos", "robust", "kf-cmp", "figures"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    config = load_config(args.config)
    p = paths(config)

    factor_data = evalue_results = None
    cached_univ = p["processed"] / "factor_universe.csv"
    if args.stage in ("all", "evalue", "oos") and cached_univ.exists():
        factor_data = pd.read_csv(cached_univ, index_col=0,
                                  parse_dates=True)

    if args.stage in ("all", "data"):
        factor_data = load_data(config, p)

    if args.stage in ("all", "evalue"):
        if factor_data is None:
            factor_data = load_data(config, p)
        evalue_results = stage_evalue(factor_data, config, p)

    if args.stage in ("all", "oos"):
        if evalue_results is None and (p["results"] / "evalue_results.csv").exists():
            evalue_results = pd.read_csv(p["results"] / "evalue_results.csv",
                                         index_col=0)
        if factor_data is None:
            factor_data = load_data(config, p)
        stage_oos(factor_data, evalue_results, config, p)

    if args.stage in ("all", "figures"):
        stage_figures(config, p, factor_data)

    if args.stage in ("all", "robust"):
        if factor_data is None:
            factor_data = load_data(config, p)
        stage_robust(factor_data, config, p)

    if args.stage == "kf-cmp":
        stage_kf_comparison(config, p)


if __name__ == "__main__":
    main()
