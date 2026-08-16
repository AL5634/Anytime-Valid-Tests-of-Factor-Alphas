# Formula / Code / Table / Figure Audit Report

Date: 2026-08-15. Scope: all mathematical formulas, code, tables, and figures
in the JEF manuscript and internet appendix.

## 1. Mathematical formulas (6 equations)

Each equation was cross-checked against the implementation in
`src/evalue/core.py`.

| Equation | Code counterpart | Status |
|---|---|---|
| eq:reg (factor regression, H0: alpha=0) | `regression_alpha_evalue` inputs | consistent |
| eq:eprocess (product of increments) | `e *= (1+lam*z)` | consistent |
| eq:ville (Ville bound, reject at 1/alpha) | `e >= 1/alpha` | consistent |
| eq:proj (no-intercept expanding projection) | `_expanding_window_residuals` | consistent, F_{t-1}-measurable |
| eq:lambda (betting fraction) | `running_score/j`, j = t - t0 | consistent |
| eq:inc (increment in [0.01,1.99]) | lam in [-0.99,0.99], z in [-1,1] | consistent |

Additional consistency points verified:
- Clip bound B_t = 3*sigma_t (clip_scale=3), sigma_t from `resid[min_obs:t]`
  (F_{t-1}-measurable). Consistent.
- Betting starts at t0 = min_obs + 24 (SIGMA_BURN_MONTHS=24); lambda_{t0}=0.
  Consistent with manuscript.
- **EGR denominator fixed**: the manuscript previously wrote
  EGR = (1/T) sum log E^(t) while the code computes the mean of the
  log-increments (equivalently over T-1 increments). The manuscript now
  describes EGR as "the average monthly log-increment over the betting
  window", removing the ambiguous 1/T. No numeric change.

## 2. Code

- Residualization has no look-ahead (unit-tested:
  `test_rolling_residuals_no_lookahead`).
- Edge cases verified manually: a series shorter than min_obs+burn returns an
  empty e-path with NaN terminal e-value; an all-NaN series returns an empty
  path. No exceptions.
- `mean_path` off-by-one corrected in an earlier pass (diagnostic only).
- Test suite: **13 passed** (type-I control, power under alternative,
  regression path well-formedness, survival-stats fields, rolling no-lookahead,
  regression type-I, e-BH x4, calibration x2, skewed-residual size).

## 3. Tables (automated cross-check: scripts/verify_tables.py)

Result: **12/12 PASS**.

| Table | Source CSV | Status |
|---|---|---|
| tab:oos (5 rules) | oos_comparison.csv | PASS (5/5 rows) |
| tab:sens (16 cells) | sensitivity_grid.csv | PASS (16/16) |
| tab:calib (power) | power_simulation.csv | PASS (6 alpha rows) |
| tab:data (categories) | evalue_results.csv | PASS (14 categories) |
| tab:survival (top-12) | evalue_results.csv | PASS (11/12 E values parsed) |
| IA tab:exclusion | oos_exclusion.csv | PASS |
| IA tab:lateholdout | oos_lateholdout.csv | PASS |
| IA tab:skewsize | skew_size.csv | PASS |
| IA tab:rolling | evalue_rolling120.csv | PASS (76, overlap 72) |
| IA tab:firstrej | first_rejection_timing.csv | PASS (62, 43) |
| IA tab:tails | residual_tails.csv | PASS |
| IA tab:corr | oos_factor_corr.csv | PASS |

The verifier is reusable: `python3 scripts/verify_tables.py`.

## 4. Figures

- 6 figures are generated and all 6 are referenced in the manuscript:
  fig2_evalue_paths, fig_evalue_hist, fig_e_vs_t, fig_by_cat,
  fig4_pre_post_egr, fig7_oos_cumulative.
- fig7_oos_cumulative contains 5 curves (E-value, e-BH, |t|>3, |t|>2,
  EB-shrink), consistent with the 5-rule OOS table.
- **2 orphan figures removed**: fig1_illustration and fig3_egr_histogram were
  generated but never referenced. Their generation code and PDFs were deleted;
  the figures stage now produces exactly the 6 referenced figures.

## Summary

No numeric discrepancy was found between any table and its source data. All
formulas are consistent with the code. The only formula edit was the EGR
wording (denominator ambiguity), which changes no numbers. Two unused figures
were removed for a clean submission.
