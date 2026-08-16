"""
construct_factors.py — Rebuild characteristic factors from raw stock data.

SCAFFOLD. Requires WRDS (CRSP + Compustat) data — not needed for the
headline analysis, which uses the publicly available published factor
returns (Open Source AP / Ken French / JKP).

Implements the standard Fama-French approach:
    - NYSE breakpoints
    - value-weighted quintile portfolios
    - long-short (top - bottom) monthly factor returns
    - microcap control (exclude stocks below the NYSE 20th percentile
      where documented).

This file is intentionally minimal; extend per characteristic when
WRDS access is available.
"""
from __future__ import annotations

import pandas as pd


def nyse_breakpoints(value: pd.Series, factor_col: str, weight_col: str):
    """Return NYSE-only cross-sectional breakpoints for a characteristic."""
    nyse = value[value.get("exchange", "") == "NYSE"] \
        if "exchange" in value.columns else value
    q25 = nyse[factor_col].quantile(0.2)
    q80 = nyse[factor_col].quantile(0.8)
    return q25, q80


def assign_quintiles(df: pd.DataFrame, factor_col: str, q_lo: float, q_hi: float):
    """Assign each stock to a quintile based on NYSE cutoffs {q_lo, q_hi}."""
    out = df[factor_col].copy()
    out = out.apply(
        lambda v: ("low" if v <= q_lo else ("high" if v >= q_hi else "mid"))
    )
    return out


def build_long_short(data: pd.DataFrame, factor_col: str,
                     ret_col: str, weight_col: str) -> pd.Series:
    """
    Build a monthly long-short factor return, value-weighted, with NYSE
    breakpoints and microcap exclusion. Returns a monthly Series.

    Parameters
    ----------
    data : long-form panel, indexed by (date, permno), columns include
           factor_col (characteristic), ret_col, weight_col, and 'exchange'.
           'mktcap' = price * shares for weighting.
    """
    rows = {}
    for date, grp in data.groupby(level="date"):
        q_lo, q_hi = nyse_breakpoints(grp, factor_col, weight_col)
        g = grp.copy()
        # microcap exclusion
        if "mktcap" in g.columns:
            nyse_cap_thr = g.loc[g["exchange"] == "NYSE", "mktcap"].quantile(0.2)
            g = g[g["mktcap"] >= nyse_cap_thr]
        g["quintile"] = assign_quintiles(g, factor_col, q_lo, q_hi)
        long = g[g["quintile"] == "high"]
        short = g[g["quintile"] == "low"]
        w_long = long[weight_col] / long[weight_col].sum()
        w_short = short[weight_col] / short[weight_col].sum()
        r_long = float((w_long * long[ret_col]).sum())
        r_short = float((w_short * short[ret_col]).sum())
        rows[date] = r_long - r_short
    return pd.Series(rows).sort_index()
