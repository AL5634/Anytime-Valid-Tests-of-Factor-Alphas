"""
cz.py — Chen-Zimmermann (Open Source Asset Pricing) factor universe loader.

Loads the 212 predictor long/short portfolio returns of Chen & Zimmermann
(2022, Critical Finance Review) from a LOCAL copy of the authors' wide
release file (`PredictorLSretWide.csv`), caches the processed panel
locally, and exposes publication-year / category metadata parsed from the
SignalDoc.csv catalogue shipped in the cloned OSAP repo.

The wide file is the "Predictor" long/short release: one column per
signal (SignalDoc Acronym), returns in PERCENT. We convert to decimal
monthly returns and normalise the date column to month-end. No
openassetpricing / Google-Drive dependency is required.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CZ_CACHE = PROJECT_ROOT / "data" / "processed"
CZ_PANEL_FILE = CZ_CACHE / "cz_universe.csv"
CZ_META_FILE = CZ_CACHE / "cz_metadata.pkl"

# Candidate locations for the locally-downloaded wide LS release file.
CZ_WIDE_CANDIDATES = [
    PROJECT_ROOT / "PredictorLSretWide.csv",
    PROJECT_ROOT / "data" / "downloaded" / "PredictorLSretWide.csv",
]

# SignalDoc catalogue shipped in the cloned OSAP GitHub repo.
SIGNAL_DOC_FILE = (
    PROJECT_ROOT / "data" / "downloaded" / "CrossSection" / "SignalDoc.csv"
)

# Month-end date of the final observation in the CZ release (2023-12-29
# parses to the 2023-12 month-end). Keep the panel strictly below 2024-01.
MAX_DATE = pd.Timestamp("2023-12-31")


class CZDataUnavailable(RuntimeError):
    """Raised when the Chen-Zimmermann wide file is not found locally."""


def _find_wide_file() -> Path:
    for p in CZ_WIDE_CANDIDATES:
        if p.exists():
            return p
    raise CZDataUnavailable(
        "PredictorLSretWide.csv not found. Download it from "
        "https://www.openassetpricing.com/data and place it in the project "
        "root (or data/downloaded/).")


def load_cz_universe(min_obs: int = 60, refresh: bool = False) -> pd.DataFrame:
    """
    Load the wide Chen-Zimmermann long/short predictor panel.

    Returns a DataFrame of DECIMAL monthly returns, month-end index,
    columns = SignalDoc Acronyms, sample 1926-01 .. 2023-12. Columns with
    fewer than `min_obs` non-missing months are dropped. Cached to
    data/processed/cz_universe.csv.
    """
    if not refresh and CZ_PANEL_FILE.exists():
        return pd.read_csv(CZ_PANEL_FILE, index_col=0, parse_dates=True)

    src = _find_wide_file()
    raw = pd.read_csv(src)
    date = pd.to_datetime(raw["date"], format="%Y-%m-%d", errors="coerce")
    panel = raw.drop(columns=["date"]).apply(pd.to_numeric, errors="coerce")

    # normalise to month-end index; the release dates are already month-end
    idx = date + pd.offsets.MonthEnd(0)
    panel.index = idx
    panel = panel.sort_index()

    # percents -> decimals
    panel = panel / 100.0

    # keep only months within the CZ release window and factors with data
    panel = panel[panel.index <= MAX_DATE]
    panel = panel[[c for c in panel.columns if panel[c].notna().sum() >= min_obs]]

    CZ_CACHE.mkdir(parents=True, exist_ok=True)
    panel.to_csv(CZ_PANEL_FILE)
    return panel


def load_cz_metadata(refresh: bool = False) -> pd.DataFrame:
    """
    SignalDoc metadata (from the locally-cloned OSAP repo) for the CZ
    universe: publication year, authors, economic category, signal type,
    and the documented sign. Cached to data/processed/cz_metadata.pkl.
    """
    if not refresh and CZ_META_FILE.exists():
        with open(CZ_META_FILE, "rb") as f:
            return pickle.load(f)
    if not SIGNAL_DOC_FILE.exists():
        raise CZDataUnavailable(
            f"SignalDoc.csv not found at {SIGNAL_DOC_FILE}. Clone the OSAP "
            "repo or re-download it to restore metadata.")
    doc = pd.read_csv(SIGNAL_DOC_FILE)
    meta = doc[["Acronym", "Authors", "Year", "Cat.Economic", "Cat.Signal",
                "Sign"]].rename(
                    columns={"Acronym": "acronym",
                             "Cat.Economic": "cat_economic",
                             "Cat.Signal": "cat_signal",
                             "Sign": "sign"})
    meta["pub_year"] = pd.to_numeric(meta["Year"], errors="coerce").astype("Int64")
    meta = meta.drop(columns=["Year"])
    CZ_CACHE.mkdir(parents=True, exist_ok=True)
    with open(CZ_META_FILE, "wb") as f:
        pickle.dump(meta, f)
    return meta


def cz_publication_dates(meta: pd.DataFrame) -> dict[str, str]:
    """Map factor Acronym -> publication date 'YYYY-01-31' (month-end of the
    publication year, matching the month-end panel index). Factors without a
    known year are excluded."""
    out = {}
    for _, row in meta.iterrows():
        yr = row.get("pub_year")
        if pd.notna(yr):
            out[row["acronym"]] = f"{int(yr)}-01-31"
    return out
