# Anytime-Valid Tests of Factor Alphas — Replication Code and Data

Replication package for "Anytime-Valid Tests of Factor Alphas": an
anytime-valid (e-process) test of the intercept in factor time-series
regressions, applied to the 212 public long--short predictors of Chen and
Zimmermann (2022) against the Fama--French five-factor model.

## 1. Contents

```
├── config.yaml                 # all tunable parameters
├── Makefile                    # one-command reproducibility
├── requirements.txt
├── pyproject.toml
├── src/
│   ├── config.py               # config loader
│   ├── run.py                  # end-to-end runner
│   ├── data/                   # data download + processing
│   ├── evalue/                 # e-value / e-process core
│   ├── analysis/               # OOS horse race, robustness, diagnostics
│   └── viz/                    # figures
├── tests/                      # unit tests (Type-I + power + diagnostics)
├── data/downloaded/            # frozen public data vintage (committed)
│   ├── ff5_panel.csv           # Fama-French 5 factors (frozen 2026-08-16)
│   └── CrossSection/SignalDoc.csv  # factor metadata (categories, pub years)
└── output/figures/             # paper figures (committed)
```

`data/processed/` and `data/results/` are populated at runtime and are
gitignored.

## 2. Data

The analysis uses only public data (no WRDS subscription):

- **CZ predictor returns**: `PredictorLSretWide.csv` (place it in the
  project root), the public "Predictor" long--short wide release from
  openassetpricing.com (values in percent, converted to decimals in the
  pipeline). Download instructions and the license are in
  `data/downloaded/README.md`.
- **Fama--French five factors** (MKT, SMB, HML, RMW, CMA, RF): downloaded
  from the Ken French Data Library
  (`F-F_Research_Data_5_Factors_2x3_TXT.zip`) and **frozen** in
  `data/downloaded/ff5_panel.csv` on 2026-08-16 (756 months,
  1963:07--2026:06) so every rerun uses the same data vintage; the
  pipeline reads the frozen file first and only falls back to a live
  download if it is absent.
- **Factor metadata**: `data/downloaded/CrossSection/SignalDoc.csv`
  (acronyms, economic categories, publication years) from the authors'
  OSAP GitHub repository.

Sample: 1963:07--2023:12 intersection (726 months). The universe is
assembled by `src/data/cz.py::load_cz_universe` plus the FF5 panel.

A Ken French 16-factor subset (momentum, reversal, size, value,
profitability, investment, accruals, beta, issuance, volatility) is
assembled by `src/data/download.py::load_factor_universe` and used only
for the appendix comparison (`--stage kf-cmp`).

## 3. Installation

```bash
cd "Anytime-Valid Tests of Factor Alphas"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Running

```bash
make data       # download / build the factor panel
make evalue     # compute e-values for all factors  -> data/results/evalue_results.csv
make oos        # OOS horse race                    -> data/results/oos_comparison.csv
make figures    # figures                           -> output/figures/
make test       # unit tests
make all        # data + evalue + oos + figures
```

## 5. Configuration

Edit `config.yaml`:

- `evalue.alpha` (0.05), `evalue.threshold` (20 = 1/alpha);
- `evalue.min_obs` (60 months);
- `empirical.*` train/test split (default `1963-07`-`2005-12` /
  `2006-01`-`2023-12` (CZ release end));
- `empirical.factor_model` (`CAPM` | `FF3` | `FF5`);
- `screening.*` thresholds for the OOS horse race (e-value / HLZ
  `|t|>3` / empirical-Bayes shrinkage / `|t|>2`).

## 6. Methodology (summary)

For each factor we test `H0: alpha = 0` in the time-series regression

    r_F,t = alpha + beta' FF_t + epsilon_t

using a **residualize-and-bet** anytime-valid e-process:

1. **Residualize** — OLS-project the factor returns on the FF factor
   loadings *without* an intercept, estimated on an expanding window, so
   the residuals are orthogonal to the factors but their mean is `alpha`
   (not forced to zero).
2. **Bet** — run a bounded-betting test martingale on the residuals:
   clip to `[-1,1]` with `B = 3*sigma`, choose the predictable
   empirical-Bayes betting fraction `lambda_t`, and form
   `E^(t) = 1 + lambda_t Z_t`.

Because `lambda_t` is past-measurable and `E[Z_t | past] = 0` under
the null, the running product `E_T = prod E^(t)` is a nonnegative
supermartingale: an e-process satisfying
`P(H0){ exists t : E_t >= 1/alpha } <= alpha` (Ville), valid under
**arbitrary optional stopping / continuation**.

Implementations:

- `src/evalue/core.py::_betting_egress_process` — bounded-betting
  e-process (Waudby-Smith & Ramdas 2021 / Darling-Robbins).
- `src/evalue/core.py::regression_alpha_evalue` — residualize-and-bet
  e-process for the intercept, returning the path, terminal e-value,
  first rejection date, and EGR.
- `src/evalue/core.py::factor_survival_stats` — diagnostics.
- `src/evalue/batch.py::compute_all_factor_evals` — batch computation,
  splitting each path at the publication date for the pre/post-publication
  life-cycle diagnostic.
- `src/analysis/oos_horse_race.py` — OOS horse race: E-value vs. HLZ
  (`|t|>3`) vs. empirical-Bayes shrinkage vs. `|t|>2`, with an optional
  count-equalized comparison.

## 7. Code availability

The complete data construction pipeline and analysis code are publicly
available on GitHub:
https://github.com/AL5634/Reflexive-Learning-in-Asset-Pricing-Estimating-Anomaly-Erosion-and-Its-Portfolio-Implications.git
