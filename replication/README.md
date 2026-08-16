# Betting on Factors: Anytime-Valid Inference for the Factor Zoo


## 1. Contents

```
evalue_factor_zoo/
├── config.yaml                 # all tunable parameters
├── Makefile                    # one-command reproducibility
├── requirements.txt
├── src/
│   ├── config.py               # config loader
│   ├── run.py                  # end-to-end runner
│   ├── data/                   # data download + processing
│   ├── evalue/                 # e-value / e-process core
│   ├── analysis/               # main, OOS horse race, robustness
│   └── viz/                    # figures
├── tests/                      # unit tests (Type-I + power + diagnostics)
├── data/                       # (populated at runtime; gitignored raw)
├── output/                     # tables + figures

```

## 2. Data

The headline analysis is the **212-predictor Chen–Zimmermann (2022)
long–short universe** conditioned on the Fama–French five-factor model,
using only public data (no WRDS subscription):

- **CZ predictor returns**: `PredictorLSretWide.csv` (project root),
  the public "Predictor" long–short wide release from
  openassetpricing.com (values in percent, converted to decimals in the
  pipeline). Download instructions and the license are in
  `data/downloaded/README.md`.
- **Fama–French five factors** (MKT, SMB, HML, RMW, CMA, RF): downloaded
  from the Ken French Data Library
  (`F-F_Research_Data_5_Factors_2x3_TXT.zip`) and **frozen** in
  `data/downloaded/ff5_panel.csv` on 2026-08-16 (756 months,
  1963:07–2026:06) so every rerun uses the same data vintage; the
  pipeline reads the frozen file first and only falls back to a live
  download if it is absent.
- **Factor metadata**: `data/downloaded/CrossSection/SignalDoc.csv`
  (acronyms, economic categories, publication years) from the authors'
  OSAP GitHub repository.

Sample: 1963:07–2023:12 intersection (726 months). The universe is
assembled by `src/data/cz.py::load_cz_universe` plus the FF5 panel.

A Ken French 16-factor subset (momentum, reversal, size, value,
profitability, investment, accruals, beta, issuance, volatility) is
assembled by `src/data/download.py::load_factor_universe` and used only
for the appendix comparison (`--stage kf-cmp`).

