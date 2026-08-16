"""
test_core.py — Unit tests for the e-value core.

Verifies:
  1. The betting e-process is a martingale / any time-valid under the
     null (mean 0): e-values do not drift up; Type-I error <= alpha.
  2. Under a persistent positive-drift alternative, the e-process grows
     and eventually crosses the 1/alpha threshold.
  3. `regression_alpha_evalue` returns an interpretable path and the
     diagnostics are well-formed.
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evalue.core import anytime_mean_cs, regression_alpha_evalue, \
    factor_survival_stats


def _empirical_type1(reps=500, n=200, sigma=0.05, alpha=0.05, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    rej = 0
    for _ in range(reps):
        x = rng.normal(0, sigma, n)
        res = anytime_mean_cs(pd.Series(x), alpha=alpha)
        if (res["e_path"] >= res["threshold"]).any():
            rej += 1
    return rej / reps


def test_type1_control():
    """Under H0 (mean=0) empirical rejection rate must be <= alpha.

    Monte-Carlo margin: nominal alpha = 0.05 with reps = 500 gives
    s.e. ~= sqrt(0.05*0.95/500) ~= 0.0097; we allow 2 s.e. above nominal
    (the e-process is much more conservative in practice).
    """
    rate = _empirical_type1()
    mc_se = (0.05 * 0.95 / 500) ** 0.5
    bound = 0.05 + 2 * mc_se
    assert rate <= bound, f"Type-I rate {rate:.3f} too high under H0 (bound {bound:.3f})"


def test_power_under_alternative():
    """A strong persistent drift (e.g. momentum-like, ~1%/month) must
    generate an e-process that crosses the 1/alpha threshold within a
    realistic horizon. Anytime-valid tests are conservative, so weak
    effects need longer samples -- this is a feature, not a bug."""
    rng = np.random.default_rng(3)
    n = 2000
    x = pd.Series(0.01 + rng.normal(0, 0.04, n))  # ~1%/month alpha
    res = anytime_mean_cs(x, alpha=0.05)
    evalue = res["e_path"].dropna()
    assert evalue.iloc[-1] > res["threshold"], \
        "e-process did not reject under a persistent alternative"


def test_regression_alpha_path_wellformed():
    """regression_alpha_evalue returns well-formed outputs."""
    rng = np.random.default_rng(5)
    idx = pd.date_range("1965-01-01", periods=400, freq="MS")
    ff = pd.DataFrame({
        "MKT": rng.normal(0.006, 0.05, 400),
        "SMB": rng.normal(0.002, 0.03, 400),
        "HML": rng.normal(0.002, 0.03, 400),
    }, index=idx)
    returns = pd.Series(0.003 + 0.5 * ff["MKT"] - 0.2 * ff["SMB"]
                        + rng.normal(0, 0.02, 400), index=idx)
    res = regression_alpha_evalue(returns, ff, min_obs=60)
    assert res["e_path"] is not None
    assert res["evalue_final"] == res["evalue_final"]  # not nan


def test_survival_stats_fields():
    rng = np.random.default_rng(7)
    idx = pd.date_range("1970-01-01", periods=300, freq="MS")
    e = 1.0 + np.arange(300) * 0.1 + rng.normal(0, 1, 300)
    evalue_path = pd.Series(np.cumprod(np.clip((e / e), 1e-12, np.inf))
                            * (1.01 ** np.arange(300)), index=idx)
    stats = factor_survival_stats(evalue_path, publication_date="2000-01-01")
    assert "evalue_final" in stats
    assert "egr" in stats
    assert "pre_pub_egr" in stats


def test_rolling_residuals_no_lookahead():
    from src.evalue.core import _rolling_window_residuals
    rng = __import__('numpy').random.default_rng(0)
    import numpy as np
    n, p, win = 300, 5, 120
    y = rng.normal(size=n)
    ff = rng.normal(size=(n, p))
    resid = _rolling_window_residuals(y, ff, min_obs=60, window=win)
    # residual at time t must not depend on y[t:] or ff[t:].
    y2 = y.copy(); ff2 = ff.copy()
    t = 250
    y2[t+1:] = rng.normal(size=n-t-1)
    ff2[t+1:] = rng.normal(size=(n-t-1, p))
    resid2 = _rolling_window_residuals(y2, ff2, min_obs=60, window=win)
    assert np.allclose(resid[:t+1], resid2[:t+1], equal_nan=True)


def test_regression_alpha_type1_control():
    """Under H0 (alpha=0) the regression-alpha e-process must control the
    anytime-valid type-I error: the empirical probability of the path ever
    crossing 1/alpha must be at or below nominal alpha (within MC error)."""
    import numpy as np
    rng = np.random.default_rng(11)
    reps, n, alpha = 200, 300, 0.05
    idx = pd.date_range("1960-01-01", periods=n, freq="MS")
    rej = 0
    for _ in range(reps):
        ff = pd.DataFrame({
            "MKT": rng.normal(0.006, 0.05, n),
            "SMB": rng.normal(0.002, 0.03, n),
            "HML": rng.normal(0.002, 0.03, n),
        }, index=idx)
        r = (0.5 * ff["MKT"] - 0.3 * ff["SMB"] + 0.2 * ff["HML"]
             + rng.normal(0, 0.03, n))
        r = pd.Series(r, index=idx)
        res = regression_alpha_evalue(r, ff, min_obs=60, clip_scale=3.0,
                                      alpha=alpha)
        if not res["e_path"].empty and (res["e_path"] >= 1.0 / alpha).any():
            rej += 1
    rate = rej / reps
    mc_se = (0.05 * 0.95 / reps) ** 0.5
    bound = 0.05 + 3 * mc_se
    assert rate <= bound, f"Type-I {rate:.3f} exceeds bound {bound:.3f}"


def test_regression_alpha_size_skewed_residuals():
    """Size under a skewed residual DGP (skew-normal, skew=-0.18, matching
    the median rejector skewness): empirical anytime rejection rate must stay
    at or below nominal alpha (within MC error)."""
    import numpy as np
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.analysis.skew_size import _skewed_normal, _standardize, _one_run
    rng = np.random.default_rng(23)
    n, alpha = 300, 0.05
    reps = 300
    idx = pd.date_range("1960-01-01", periods=n, freq="MS")
    rngf = np.random.default_rng(24)
    ff = pd.DataFrame({
        "MKT": rngf.normal(0.006, 0.05, n),
        "SMB": rngf.normal(0.002, 0.03, n),
        "HML": rngf.normal(0.002, 0.03, n),
    }, index=idx)
    sigma = 0.03
    rej = 0
    for _ in range(reps):
        eps = sigma * _standardize(_skewed_normal(rng, -0.18, n))
        r = (0.5 * ff["MKT"] - 0.3 * ff["SMB"] + 0.2 * ff["HML"]
             + pd.Series(eps, index=idx))
        if _one_run(r, ff):
            rej += 1
    rate = rej / reps
    mc_se = (0.05 * 0.95 / reps) ** 0.5
    assert rate <= 0.05 + 3 * mc_se, f"size {rate:.3f} too high"
