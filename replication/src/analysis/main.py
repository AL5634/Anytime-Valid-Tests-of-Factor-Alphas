"""
main.py — Main empirical analysis (Table 1, Figs 2-6).

Produces the core deliverables:
  - Table 1 : E-value factor survival table (top factors)
  - Fig 2   : E-process paths for representative factors
  - Fig 3   : EGR distribution by factor theme
  - Fig 4   : pre- vs post-publication EGR scatter
  - Fig 5   : e-value half-life distribution
  - Fig 6   : global vs US e-value scatter  (cross-market)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import paths
from ..evalue.cross_market import compute_all_factor_evals, \
    compute_cross_market_evalue


def build_survival_table(evalue_results: pd.DataFrame, top_k: int = 50) -> pd.DataFrame:
    """Assemble Table 1: top-K factors by final e-value."""
    df = evalue_results.sort_values("evalue_final", ascending=False).head(top_k)
    return df[["evalue_final", "first_rejection_date", "egr", "max_evalue"]]


def analyze_main(factor_data: pd.DataFrame,
                 ff: pd.DataFrame,
                 config: dict,
                 remember: bool = True) -> dict:
    """Run the full main analysis pipeline."""
    p = paths(config)
    emp = config["empirical"]
    fm = emp["factor_model"]

    ff_use = ff[[c for c in pp_factors(fm, emp)]]

    evalue_results = compute_all_factor_evals(factor_data, ff_use)

    if remember:
        evalue_results.to_csv(p["results"] / "evalue_results.csv")

    # Representative factors for the e-process path figure (Fig 2)
    reps = ["mom", "hml", "smb", "rmw", "cma", "ivol"]
    reps = [r for r in reps if r in factor_data.columns][:8]

    tab1 = build_survival_table(evalue_results)
    tab1.to_csv(p["tables"] / "table1_survival.csv")

    return {
        "evalue_results": evalue_results,
        "tab1": tab1,
        "rep_factors": reps,
    }


def pp_factors(fm: str, emp: dict) -> list[str]:
    return emp[f"{fm.lower()}_factors"]
