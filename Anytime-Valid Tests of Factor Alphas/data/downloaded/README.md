# Data for the 212-predictor zoo

The factor-return panel used in this paper is the public long/short
predictor release of Chen & Zimmermann (2022, *Critical Finance Review*):

> **PredictorLSretWide.csv** (wide monthly long/short returns)

Download it from the authors' data page:

- https://www.openassetpricing.com/data
  (look for "PredictorLSretWide.csv" in the monthly long/short zip)

Place the CSV file in the **project root** (the same directory as
`config.yaml` and `Makefile`). The pipeline reads it from either:

1. `PredictorLSretWide.csv` (project root), or
2. `data/downloaded/PredictorLSretWide.csv` (data subdirectory)

After placing the file, run:

```bash
make all
```

This will:
1. Build the 212-factor panel (`src/data/cz.py` reads the local CSV,
   converts returns from percent to decimal, and aligns with the
   Fama-French five factors from Ken French's site).
2. Compute anytime-valid e-values for all 212 predictors (FF5 benchmark).
3. Run the out-of-sample horse race (train 1963:07--2005:12, test
   2006:01--2023:12).
4. Generate the power simulation and sensitivity grid.
5. Produce all figures.

