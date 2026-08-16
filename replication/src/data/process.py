"""
process.py — Data processing and harmonisation.

Normalises all factor / FF-return series to a common format:
    - month-end datetime index
    - returns in decimal (percent/100)
    - aligned onto the intersection of valid months
    - excess returns where required (minus risk-free where applicable)
"""
from __future__ import annotations

import pandas as pd


def align_monthly(series_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """Concatenate dict of aligned monthly series into a wide DataFrame."""
    out = pd.DataFrame(series_dict).dropna(how="all")
    # month-end standardisation
    out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated()]
    return out.sort_index()


def to_excess(df: pd.DataFrame, rf: pd.Series, risk_free: bool = False) -> pd.DataFrame:
    """Subtract the risk-free rate from all columns (optionally))."""
    if not risk_free:
        return df
    rf = rf.reindex(df.index).ffill()
    return df.sub(rf, axis=0)


def make_factor_metadata(factor_data: pd.DataFrame) -> pd.DataFrame:
    """
    Build a metadata frame: start date, end date, n_months, sample length,
    and (optionally) publication year per factor (to be enriched from the
    OSAP source paper metadata in a later stage).
    """
    rows = []
    for col in factor_data.columns:
        s = factor_data[col].dropna()
        rows.append({
            "factor": col,
            "start": s.index.min(),
            "end": s.index.max(),
            "n_months": len(s),
        })
    return pd.DataFrame(rows)


def align_factor_and_ff(factor_data: pd.DataFrame,
                        ff_factors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align factor panel and FF factors onto the common month index."""
    common = factor_data.index.intersection(ff_factors.index)
    return factor_data.loc[common], ff_factors.loc[common]
