"""
batch.py — Batch e-value computation for factor panels.
"""
from __future__ import annotations

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .core import regression_alpha_evalue, factor_survival_stats


def compute_all_factor_evals(
    factor_data: pd.DataFrame,
    ff_factors: pd.DataFrame,
    n_jobs: int = -1,
    verbose: bool = True,
    publication_dates: dict[str, str] | None = None,
    **ev_kwargs,
) -> pd.DataFrame:
    """
    Batch-compute e-values and survival stats for all factors.

    Parameters
    ----------
    factor_data : wide panel, index=date, columns=factor, values=return
    ff_factors  : FF factor returns on same index
    n_jobs      : parallel workers
    publication_dates : optional {factor: 'YYYY-MM-DD'} used to split each
        e-process path into pre-/post-publication segments (life-cycle
        diagnostics). Factors absent from the map are left with NaN.
    Returns
    -------
    DataFrame: one row per factor with evalue_final, first_rejection_date,
               egr, and publication-relative diagnostics.
    """
    def _one(factor: str) -> dict:
        res = regression_alpha_evalue(
            factor_data[factor], ff_factors, **ev_kwargs
        )
        pub = None
        if publication_dates:
            pub = publication_dates.get(factor)
        stats = factor_survival_stats(
            res["e_path"], publication_date=pub,
            threshold=1.0 / ev_kwargs.get("alpha", 0.05)
        )
        stats["factor_name"] = factor
        stats.update({k: res[k] for k in ("egr", "evalue_final")})
        return stats

    cols = list(factor_data.columns)
    it = cols if not verbose else tqdm(cols, desc="e-values")
    results = Parallel(n_jobs=n_jobs)(delayed(_one)(f) for f in it)
    return pd.DataFrame(results).set_index("factor_name")
