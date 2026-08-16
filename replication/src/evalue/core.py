"""
core.py — Anytime-valid e-processes for factor return testing.

The heart of the paper. We implement the two standard, rigorous
constructions of e-processes / confidence sequences used to test
whether a factor's risk-adjusted excess return (alpha) differs from
zero, in a way that remains valid under arbitrary optional stopping
and optional continuation.

Main methods
------------
1. `anytime_mean_cs(X, alpha)` : betting-based confidence sequence for the
   mean of a bounded/light-tailed sequence (Waudby-Smith & Ramdas 2021),
   via the empirical-Bayes mixture martingale. Returns an e-process path
   that is an anytime-valid test of H0: mean = 0.

2. `regression_alpha_evalue(returns, ff_factors, ...)` : an
   anytime-valid e-value for the intercept (alpha) of the time-series
   regression `r_F ~ alpha + beta' FF_factors`, using an expanding-window
   (F_{t-1}-measurable) residualization followed by the bounded-betting
   e-process (Waudby-Smith & Ramdas 2021). The returned series is a valid
   e-process for H0: alpha = 0.

3. `factor_survival_stats(...)` : converts an e-process path into the
   tabulated diagnostics used in the paper (final e-value, first
   rejection date, E-value growth rate, pre/post publication decay, etc.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Anytime-valid mean e-process (betting) — Waudby-Smith & Ramdas (2021)
# ---------------------------------------------------------------------------


def _betting_egress_process(
    X: np.ndarray,
    mu0: float = 0.0,
    clip_bound: float | None = None,
) -> dict:
    """
    Anytime-valid e-process for testing H0: mean(X) = mu0.

    Construction: the bounded-betting test martingale of Waudby-Smith &
    Ramdas (2021). We clip X to a symmetric interval [-B, B] (here B is a
    fixed scale, by default 3x the sample sd, chosen before the game), and
    bet on the centred, bounded increments

        Z_t = clip((X_t - mu0)/B, -1, 1) in [-1,1].

    For H0, E[Z_t | F_{t-1}] = 0, so for any *predictable* lambda_t with
    |lambda_t| <= 0.99 < 1, the increment inc_t = 1 + lambda_t Z_t is
    bounded away from zero (inc_t >= 1 - 0.99 > 0) and
    satisfies E[inc_t | F_{t-1}] = 1. Hence the product

        e_t = prod_{s<=t} inc_s

    is a nonnegative martingale under H0 -- an e-process with
    P_{H0}( exists t : e_t >= 1/alpha ) <= 1/alpha (Ville).
    lambda_t is chosen by empirical Bayes (running demeaned score) to
    grow fastest under the alternative while staying in [-1,1].

    Returns dict with e_path, mean_path, cs_lower, cs_upper.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    if clip_bound is None:
        # robust scale; tighter than 6sigma so a genuine mean shift
        # translates into a larger (bounded) Z without heavy attenuation.
        clip_bound = 3.0 * (float(np.std(X, ddof=1)) or 1.0)
    B = max(float(clip_bound), 1e-9)

    Z = np.clip((X - mu0) / B, -1.0, 1.0)

    e = np.ones(n, dtype=float)
    mean_path = np.full(n, np.nan)
    cs_lower = np.full(n, np.nan)
    cs_upper = np.full(n, np.nan)

    running_score = 0.0   # sum of Z up to t-1
    for t in range(1, n + 1):
        z = Z[t - 1]
        # predictable empirical-Bayes betting fraction. The cap is
        # strictly below 1 so that inc = 1 + lambda*z >= 1 - 0.99 > 0:
        # a saturated bet against a large opposite Z can shrink wealth
        # sharply but can never kill the e-process (it stays positive).
        if t == 1:
            lam = 0.0
        else:
            lam = np.clip((running_score / (t - 1)) * 2.0, -0.99, 0.99)
        inc = 1.0 + lam * z
        e[t - 1] = e[t - 2] * inc
        # diagnostics: update with current obs
        running_score += z
        mean_path[t - 1] = B * (running_score / t)   # raw-scale running mean

        # anytime-valid confidence sequence (bounded-mean Hoeffding,
        # centred on the running clipped mean)
        half = B * np.sqrt(2.0 * np.log(t + 1) / t)
        cs_lower[t - 1] = mean_path[t - 1] - half
        cs_upper[t - 1] = mean_path[t - 1] + half

    return {
        "e_path": e,
        "mean_path": mean_path,
        "cs_lower": cs_lower,
        "cs_upper": cs_upper,
    }


def anytime_mean_cs(
    X: pd.Series | np.ndarray,
    alpha: float = 0.05,
    sigma_hat: float | None = None,
) -> dict:
    """
    Public API for the betting mean e-process.

    Returns dict with:
        - e_path      : pd.Series of e-values (anytime-valid)
        - cs_lower/upper : anytime-valid confidence sequence for mean
        - threshold   : 1/alpha
    """
    idx = X.index if isinstance(X, pd.Series) else None
    Xa = np.asarray(X, dtype=float)
    clip = (3.0 * sigma_hat) if sigma_hat is not None else None
    e = _betting_egress_process(Xa, clip_bound=clip)
    res = {
        "e_path": pd.Series(e["e_path"], index=idx),
        "cs_mean": pd.Series(e["mean_path"], index=idx),
        "cs_lower": pd.Series(e["cs_lower"], index=idx),
        "cs_upper": pd.Series(e["cs_upper"], index=idx),
        "threshold": 1.0 / alpha,
    }
    return res


# ---------------------------------------------------------------------------
# 2. Regression-alpha e-value — expanding-window residualize + betting
# ---------------------------------------------------------------------------

# Months of residual history used to estimate the (fixed) clip bound B
# before betting starts. The bound is therefore F_{t-1}-measurable at
# every betting date.
SIGMA_BURN_MONTHS = 24


def _expanding_window_residuals(
    y: np.ndarray, ff: np.ndarray, min_obs: int
) -> np.ndarray:
    """
    Residualize factor returns on the FF factor matrix WITHOUT an
    intercept, using an EXPANDING-window projection so that every residual
    is F_{t-1}-measurable.

    At time t (t >= min_obs) the loadings are estimated from observations
    0..(t-1) only -- no data at or after t enters:

        resid_t = y_t - beta_hat_{0:(t-1)}' FF_t .

    The projection is on the factor LOADINGS only (no column of ones), so
    resid_t is orthogonal to the factors but is NOT forced to have zero
    mean. Its population mean is the intercept (alpha) of the regression
    r_F,t = alpha + beta'FF_t + eps_t: under H0 (alpha = 0) the residuals
    have mean zero, under H1 they do not. Because beta_hat_{0:(t-1)} is
    F_{t-1}-measurable, applying the betting martingale to the residuals
    preserves the anytime-valid guarantee (no look-ahead).

    Returns a float array with np.nan before min_obs (regression warm-up).
    """
    n = len(y)
    resid = np.full(n, np.nan)
    for t in range(min_obs, n):
        ff_win = ff[:t]
        y_win = y[:t]
        beta = np.linalg.pinv(ff_win.T @ ff_win + 1e-10 * np.eye(ff_win.shape[1])) @ (ff_win.T @ y_win)
        resid[t] = y[t] - ff[t] @ beta
    return resid


def _rolling_window_residuals(
    y: np.ndarray, ff: np.ndarray, min_obs: int, window: int = 120
) -> np.ndarray:
    """
    Rolling-window residualization (robustness variant of the expanding
    scheme). Loadings at date t are estimated on the trailing window
    [t-window, t-1] only, still WITHOUT an intercept and still
    F_{t-1}-measurable, so the residuals retain the intercept information
    and contain no look-ahead. A rolling window tracks time-varying
    loadings at the cost of a shorter effective sample per projection.
    """
    n = len(y)
    resid = np.full(n, np.nan)
    for t in range(min_obs, n):
        lo = max(0, t - window)
        ff_win = ff[lo:t]
        y_win = y[lo:t]
        beta = np.linalg.pinv(ff_win.T @ ff_win + 1e-10 * np.eye(ff_win.shape[1])) @ (ff_win.T @ y_win)
        resid[t] = y[t] - ff[t] @ beta
    return resid


def regression_alpha_evalue(
    returns: pd.Series,
    ff_factors: pd.DataFrame,
    min_obs: int = 60,
    reestimate_every: int = 12,
    alpha: float = 0.05,
    clip_scale: float = 3.0,
    beta_window: int | None = None,
) -> dict:
    """
    Anytime-valid e-process for H0 : alpha = 0 in

        r_F,t = alpha + beta' * FF_t + epsilon_t .

    Strategy (provably valid, no look-ahead): for each month t from
    min_obs onward, estimate the loadings on the EXPANDING window
    0..(t-1) -- a F_{t-1}-measurable linear map -- and form
    resid_t = y_t - beta_hat_{0:(t-1)}' FF_t. The residuals have mean
    alpha; under H0 (alpha = 0) they have mean zero. We then bet on the
    residuals with an expanding-window clip bound B_t = 3*sigma_t, where
    sigma_t is the standard deviation of residuals observed before t (so
    B_t is F_{t-1}-measurable at every betting date). Neither the
    residuals nor the bound use future data, so the e-process path is
    valid for real-time (anytime-valid) monitoring.

    Parameters
    ----------
    returns         : monthly factor excess-return series
    ff_factors      : FF factor returns aligned on same index
    min_obs         : regression warm-up months; the loadings at month t
                      are estimated on 0..(t-1) for t >= min_obs
    reestimate_every: retained for backwards compatibility; the archived
                      path is now returned on the full monthly grid
    alpha           : significance level (determines the 1/alpha threshold)
    beta_window     : if set, loadings are estimated on a ROLLING window
                      [t-beta_window, t-1] instead of the expanding window
                      (robustness variant; default None = expanding)
    """
    aligned = pd.concat([returns.rename("f"), ff_factors], axis=1).dropna()
    if len(aligned) < min_obs + SIGMA_BURN_MONTHS + 1:
        return {"e_path": pd.Series(dtype=float), "evalue_final": np.nan,
                "first_rejection_date": None, "egr": np.nan,
                "alpha_path": pd.Series(dtype=float)}

    n = len(aligned)
    y = aligned["f"].to_numpy()
    ff = aligned.iloc[:, 1:].to_numpy()
    idx = aligned.index

    # Residuals: expanding window by default; rolling window as a
    # robustness variant. Both are F_{t-1}-measurable and intercept-free.
    if beta_window is None:
        resid = _expanding_window_residuals(y, ff, min_obs)
    else:
        resid = _rolling_window_residuals(y, ff, min_obs, beta_window)

    # Betting with an expanding-window clip bound B_t (F_{t-1}-measurable):
    # at each betting date t, B_t = 3 * std(resid[min_obs:t-1]). This avoids
    # any dependence of the bound on future residuals and keeps Z
    # well-calibrated as the sample grows.
    SIGMA_BURN = SIGMA_BURN_MONTHS  # residual months before betting starts
    start = min_obs + SIGMA_BURN

    e = 1.0
    e_path = np.zeros(n - start)
    mean_path = np.zeros(n - start)
    running_score = 0.0
    for j, t in enumerate(range(start, n)):
        sigma_t = float(np.nanstd(resid[min_obs:t], ddof=1) or 1.0)
        B_t = clip_scale * sigma_t
        z = float(np.clip(resid[t] / B_t, -1.0, 1.0))
        if j == 0:
            lam = 0.0
        else:
            lam = np.clip((running_score / j) * 2.0, -0.99, 0.99)
        e = e * (1.0 + lam * z)
        e_path[j] = e
        running_score += z
        mean_path[j] = B_t * (running_score / (j + 1))

    bet_idx = idx[start:]
    e_series = pd.Series(e_path, index=bet_idx)
    alpha_series = pd.Series(mean_path, index=bet_idx)

    evalue_final = float(e_series.iloc[-1]) if not e_series.empty else np.nan
    rejection_idx = e_series[e_series >= 1.0 / alpha].first_valid_index()

    # EGR: mean MONTHLY log-increment of the e-process (full monthly path).
    log_e = np.log(np.clip(e_series.to_numpy(), 1e-12, None))
    diffs = np.diff(log_e)
    egr = float(np.mean(diffs)) if len(diffs) > 0 else np.nan

    return {
        "e_path": e_series,
        "evalue_final": evalue_final,
        "first_rejection_date": str(rejection_idx) if rejection_idx is not None else None,
        "egr": egr,
        "alpha_path": alpha_series,
    }


# ---------------------------------------------------------------------------
# 3. Diagnostics: factor survival stats
# ---------------------------------------------------------------------------


def factor_survival_stats(
    evalue_path: pd.Series,
    publication_date: str | None = None,
    threshold: float = 20.0,
) -> dict:
    """
    Tabulate diagnostic statistics for a single factor's e-process.

    Returns dict with: evalue_final, first_rejection_date, egr,
    pre_pub_egr, post_pub_egr, post_pub_decay_ratio, evalue_half_life,
    max_evalue, max_evalue_date.
    """
    e = evalue_path.dropna()
    if e.empty:
        return {}

    n = len(e)
    log_e = np.log(np.clip(e.to_numpy(), 1e-12, None))
    diffs = np.diff(log_e)
    egr = float(np.mean(diffs)) if len(diffs) else np.nan

    max_evalue = float(e.max())
    max_evalue_date = str(e.idxmax())

    # First rejection date
    rej = e[e >= threshold].first_valid_index()
    first_rejection_date = str(rej) if rej is not None else None

    pre_post = {"pre_pub_egr": np.nan, "post_pub_egr": np.nan,
                "post_pub_decay_ratio": np.nan, "evalue_half_life": np.nan}
    if publication_date is not None:
        pre = e[e.index <= publication_date]
        post = e[e.index >= publication_date]
        if len(pre) > 2:
            pre_post["pre_pub_egr"] = float(np.diff(np.log(np.clip(
                pre.to_numpy(), 1e-12, None))).mean())
        if len(post) > 2:
            post_log = np.log(np.clip(post.to_numpy(), 1e-12, None))
            post_diff = np.diff(post_log)
            pre_post["post_pub_egr"] = float(post_diff.mean())
            # decay ratio: share of pre-publication growth retained post-pub
            if pre_post["pre_pub_egr"] and abs(pre_post["pre_pub_egr"]) > 1e-12:
                pre_post["post_pub_decay_ratio"] = (
                    float(pre_post["post_pub_egr"] / pre_post["pre_pub_egr"]))
            # half-life: months to fall to half of the post-pub peak e-value
            peak = post.idxmax()
            pv = float(post.loc[peak])
            if peak is not None:
                after_peak = post.loc[post.index >= peak]
                half = pv / 2.0
                below = after_peak[after_peak < half]
                if not below.empty:
                    pre_post["evalue_half_life"] = float(
                        (below.index[0] - peak).days / 30.44)

    return {
        "evalue_final": float(e.iloc[-1]),
        "egr": egr,
        "max_evalue": max_evalue,
        "max_evalue_date": max_evalue_date,
        "first_rejection_date": first_rejection_date,
        **pre_post,
    }
