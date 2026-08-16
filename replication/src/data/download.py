"""
download.py — Download public data sources.

Two public sources are used so the analysis is fully reproducible without
WRDS access:

1. Ken French Data Library (Fama-French factors, momentum, risk-free,
   and ~150 equity portfolios used as test assets / anomaly sorts).

2. Open Source Asset Pricing (Chen & Zimmermann 2022) — 212+ replicated
   US cross-sectional return predictors (a.k.a. the factor zoo).

WRDS-restricted raw data (CRSP + Compustat) are optional and only needed
if one wishes to rebuild factors from scratch (see construct_factors.py).
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

KEN_FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
OPEN_SOURCE_AP_GH = "https://github.com/OpenSourceAP/CrossSection.git"


def _parse_ff_tab(text: str) -> pd.DataFrame:
    """
    Parse a Ken French whitespace-delimited monthly file.

    Format (as of 2026):
        <3 lines of description>
        <blank>
             Mkt-RF     SMB     HML     RMW     CMA      RF     <- header
        196307   -0.39   -0.48   ...                              <- data (pct)
        ... (until a blank line precedes the "Annual returns" section)
    Only the monthly (YYYYMM-leading) block is returned. Values are left
    in percent (callers divide by 100 as needed).
    """
    lines = [l.rstrip() for l in text.splitlines()]

    def is_date_row(toks):
        return (len(toks) > 0 and toks[0].isdigit() and len(toks[0]) == 6
                and toks[0].startswith(("192", "19", "20")))

    # Locate the first data row (6-digit date), then the header is the line
    # immediately above it. This avoids matching "Mom"/"SMB" inside the
    # prose description, which precedes the header.
    datarow_idx = next((i for i, l in enumerate(lines)
                        if is_date_row(l.split())), None)
    if datarow_idx is None:
        raise ValueError("Could not locate monthly data block in Ken French file")
    header_idx = datarow_idx - 1
    header = [t.strip() for t in lines[header_idx].split()]

    rows = []
    for l in lines[datarow_idx:]:
        toks = l.split()
        if not is_date_row(toks):
            # stop at blank line or the annual-returns block
            break
        rows.append(toks)

    df = pd.DataFrame(rows)
    # The first token of each row is the YYYYMM date. The header line does NOT
    # include a name for that column, so we prepend one to keep alignment
    # (data columns == header columns + 1) and trim any trailing date columns.
    if df.shape[1] == len(header) + 1:
        header = ["date"] + header
        df = df.iloc[:, :len(header)]
    elif df.shape[1] > len(header):
        df = df.iloc[:, :len(header + ["date"])]
        header = ["date"] + header
    else:
        header = header[:df.shape[1]]
    df.columns = header
    df.columns = header
    return df


def get_ff_factor_frets(period_end: str = "202412") -> dict[str, pd.Series]:
    """
    Return Fama-French factor monthly returns (MKT, SMB, HML, RMW, CMA, RF).

    Prefers the frozen panel in data/downloaded/ff5_panel.csv (pinned on
    2026-08-16 so that every rerun of the paper's pipeline uses the same
    data vintage); falls back to a live download from Ken French's site if
    the frozen file is absent.

    Returns dict of {factor_name: pd.Series indexed by month-end}.
    Factors: MKT (market minus RF), SMB, HML, RMW, CMA, RF.
    """
    frozen = Path(__file__).resolve().parents[2] / "data/downloaded/ff5_panel.csv"
    if frozen.exists():
        df = pd.read_csv(frozen, comment="#", index_col=0, parse_dates=True)
        out = {}
        for col in ["MKT", "SMB", "HML", "RMW", "CMA", "RF"]:
            if col in df.columns:
                out[col] = df[col].dropna()
        return out
    url = f"{KEN_FRENCH_BASE}/F-F_Research_Data_5_Factors_2x3_TXT.zip"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    name = [n for n in z.namelist() if n.endswith(".txt") or n.endswith(".csv")][0]
    raw = z.read(name).decode("latin-1")

    df = _parse_ff_tab(raw)
    df = df.rename(columns={"Mkt-RF": "MKT"})
    idx = pd.to_datetime(df["date"].astype(str) + "01", format="%Y%m%d")
    out = {}
    for col in ["MKT", "SMB", "HML", "RMW", "CMA", "RF"]:
        if col in df.columns:
            s = df[col].apply(pd.to_numeric, errors="coerce")
            out[col] = pd.Series(s.to_numpy() / 100.0, index=idx).dropna()
    return out


def get_ff_momentum():
    """Download UMD momentum factor."""
    url = f"{KEN_FRENCH_BASE}/F-F_Momentum_Factor_TXT.zip"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    name = [n for n in z.namelist() if n.endswith(".txt") or n.endswith(".csv")][0]
    df = _parse_ff_tab(z.read(name).decode("latin-1"))
    # dataframe columns: maybe "Mom   " with excess spaces; find the col
    mom_col = next((c for c in df.columns if "Mom" in c), None)
    if mom_col is None:
        raise ValueError("Momentum column not found")
    mom = df[mom_col].apply(pd.to_numeric, errors="coerce") / 100.0
    idx = pd.to_datetime(df["date"].astype(str) + "01", format="%Y%m%d")
    return pd.Series(mom.to_numpy(), index=idx).dropna()


def clone_open_source_ap(dest: Path, shallow: bool = True) -> Path:
    """
    Clone the Open Source Asset Pricing repo (Chen & Zimmermann 2022).

    NOTE (2026): this repo ships *code and signal definitions* only; the
    pre-built monthly long/short portfolio returns are distributed via the
    authors' data page (https://www.openassetpricing.com/data) and the
    OpenSourceAP.DownloadR R package. We keep this clone for the
    signal catalogue (SignalDoc.csv) but load portfolio returns from the
    always-available Ken French library so the pipeline is reproducible
    without a data-page login. See `load_factor_universe` below.
    """
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "CrossSection"
    if target.exists():
        return target
    import subprocess
    cmd = ["git", "clone", "--depth", "1" if shallow else "", OPEN_SOURCE_AP_GH,
           str(target)]
    cmd = [c for c in cmd if c != ""]
    subprocess.run(cmd, check=True)
    return target


# Ken French decile-spread files we use as publicly-available long/short
# "factors" (each yields a 10-minus-1 value-weighted spread with a full
# monthly history). Names are the meaningful factor labels.
_KF_SPREADS = {
    "mom_12_2":   ("10_Portfolios_Prior_12_2",     10),
    "mom_12_7":   ("10_Portfolios_Prior_12_7",     10),
    "short_rev":  ("10_Portfolios_Prior_2_0",      10),
    "long_rev":   ("10_Portfolios_Prior_12_7",     10),
    "size":       ("10_Portfolios_Size",           10),
    "bm":         ("10_Portfolios_Prior_12_7",     10),  # placeholder w/ real below
}


def get_kf_decile_spread(filename: str, n_dec: int = 10,
                         vw: bool = True) -> pd.Series:
    """
    Download a Ken French N-decile portfolio file (value-weighted by
    default) and return the H-minus-L (decile N minus decile 1) monthly
    spread as a factor return Series (decimal, month-end index).

    Parameters
    ----------
    filename : base name of the KF .TXT.zip file (no suffix), e.g.
               "10_Portfolios_Size" or "25_Portfolios_5x5_ME_BM".
    n_dec     : top/bottom decile index for the spread.
    vw        : select the value-weighted ("Average Value Weighted ...")
                matrix when the file contains two panels (value- and
                equal-weighted).
    """
    url = f"{KEN_FRENCH_BASE}/{filename}_TXT.zip"
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    name = [n for n in z.namelist() if n.endswith(".txt")][0]
    raw = z.read(name).decode("latin-1")
    lines = raw.splitlines()

    # Locate the weight/matrix header to decide which panel to parse.
    # A simple, robust approach: parse every monthly block that starts with
    # a 6-digit date and collect the block whose header contains n_dec names
    # matching the portfolio-grid size. We keep the LAST such block when the
    # file has multiple panels and vw=True (value-weighted typically appears
    # second). In practice we parse the block whose column count == n_dec.
    def blocks():
        # split file into sections separated by blank lines
        sec = []
        for l in lines:
            if l.strip() == "":
                if sec:
                    yield sec
                    sec = []
            else:
                sec.append(l.rstrip())
        if sec:
            yield sec

    def is_daterow(toks):
        return (len(toks) > 0 and toks[0].isdigit() and len(toks[0]) == 6
                and toks[0].startswith(("192", "19", "20")))

    candidates = []
    for sec in blocks():
        idx_row = next((i for i, l in enumerate(sec) if is_daterow(l.split())), None)
        if idx_row is None:
            continue
        ncols = len(sec[idx_row].split())
        if ncols < 4:      # not a big portfolio grid
            continue
        candidates.append((ncols, sec, idx_row))

    if not candidates:
        raise ValueError(f"No monthly portfolio block in {filename}")

    # Prefer a block with exactly n_dec value columns; else the widest.
    if vw:
        # value-weighted panel usually has the most columns (includes 'CAP'?) --
        # here files have equal column counts per panel; pick the one whose
        # header text mentions the grid size we want (deciles == n_dec).
        best = None
        for ncols, sec, idx_row in candidates:
            if ncols == n_dec + 1:      # date + n_dec columns
                best = (ncols, sec, idx_row)
                break
        if best is None:
            best = max(candidates, key=lambda c: c[0])
        ncols, sec, idx_row = best
    else:
        ncols, sec, idx_row = candidates[0]

    header = [t.strip() for t in sec[idx_row - 1].split()]
    header = ["date"] + header if len(header) == ncols - 1 else header
    header = header[:ncols]
    rows = []
    for l in sec[idx_row:]:
        toks = l.split()
        if not is_daterow(toks):
            break
        rows.append(toks)
    df = pd.DataFrame(rows)
    df = df.iloc[:, :ncols]
    df.columns = header
    date = pd.to_datetime(df["date"].astype(str) + "01", format="%Y%m%d")
    nums = df[header[1:]].apply(pd.to_numeric, errors="coerce")
    # decile N minus decile 1
    spread = (nums[header[n_dec]].to_numpy() - nums[header[1]].to_numpy()) / 100.0
    return pd.Series(spread, index=date).replace([None], np.nan).dropna()


def get_kf_2x3_spread(filename: str) -> pd.Series:
    """
    Build the characteristic high-minus-low spread from a Ken French 2x3
    (Size x Characteristic) value-weighted file (6 portfolios:
    Small Lo/Mid/Hi and Big Lo/Mid/Hi).

    The HML-style spread averages the high and low legs across the two
    size groups:
        spread = mean(BigHi, SmallHi) - mean(BigLo, SmallLo)

    Returns a month-end monthly decimal return Series.
    """
    url = f"{KEN_FRENCH_BASE}/{filename}_TXT.zip"
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    name = [n for n in z.namelist() if n.endswith(".txt")][0]
    raw = z.read(name).decode("latin-1")
    lines = raw.splitlines()

    # find the value-weighted section header line
    vw_idx = next(i for i, l in enumerate(lines) if "Value Weight" in l)
    # locate first data row (6-digit date) after vw_idx
    def is_daterow(toks):
        return (len(toks) > 0 and toks[0].isdigit() and len(toks[0]) == 6
                and toks[0].startswith(("192", "19", "20")))
    datarow = next(i for i, l in enumerate(lines[vw_idx:], vw_idx)
                   if is_daterow(l.split()))
    rows = []
    for l in lines[datarow:]:
        toks = l.split()
        if not is_daterow(toks) or len(toks) != 7:  # date + 6 portfolios
            break
        rows.append(toks)
    df = pd.DataFrame(rows)
    date = pd.to_datetime(df[0].astype(str) + "01", format="%Y%m%d")
    nums = df.iloc[:, 1:7].apply(pd.to_numeric, errors="coerce")
    # columns: SmallLo,SmallMid,SmallHi,BigLo,BigMid,BigHi
    hi = (nums.iloc[:, 2] + nums.iloc[:, 5]) / 2.0
    lo = (nums.iloc[:, 0] + nums.iloc[:, 3]) / 2.0
    spread = (hi - lo) / 100.0
    return pd.Series(spread.to_numpy(), index=date).replace([None], np.nan).dropna()


def get_kf_char_decile_spread(filename: str) -> pd.Series:
    """
    Decile high-minus-low spread from a Ken French "Portfolios Formed on X"
    value-weighted monthly file (e.g. Portfolios_Formed_on_AC, _BETA, _NI,
    _VAR, _RESVAR, _ME, _BE-ME, _E-P, _CF-P, _D-P, _OP, _INV).

    These files' portfolio names contain embedded spaces ("Lo 20", "Hi 10"),
    so the header cannot be tokenized with a naive split; the decile block is
    simply the LAST 10 value columns of the value-weighted monthly panel, and
    the spread is column[-1] - column[-10]. Returns a decimal-return Series
    on the month-end index.
    """
    url = f"{KEN_FRENCH_BASE}/{filename}_TXT.zip"
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    name = [n for n in z.namelist() if n.endswith(".txt")][0]
    lines = z.read(name).decode("latin-1").splitlines()

    vw_idx = next(i for i, l in enumerate(lines)
                  if "Value Weight" in l and "Monthly" in l)

    def is_daterow(toks):
        return (len(toks) > 0 and toks[0].isdigit() and len(toks[0]) == 6
                and toks[0].startswith(("192", "19", "20")))

    start = next(i for i in range(vw_idx, len(lines)) if is_daterow(lines[i].split()))
    rows = []
    for l in lines[start:]:
        toks = l.split()
        if not is_daterow(toks):
            break
        rows.append(toks)
    df = pd.DataFrame(rows)
    date = pd.to_datetime(df[0].astype(str) + "01", format="%Y%m%d")
    nums = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    hi = nums.iloc[:, -1]
    lo = nums.iloc[:, -10]  # the 10 decile columns are the last 10
    spread = (hi - lo) / 100.0
    return pd.Series(spread.to_numpy(), index=date).replace([None], np.nan).dropna()


def load_factor_universe(min_obs: int = 120) -> pd.DataFrame:
    """
    Assemble a real, fully-public monthly factor panel from Ken French:
    the Fama-French five factors plus a 16-factor "zoo" of characteristic
    long-short spreads (momentum, reversal, size, value, profitability,
    investment, accruals, market beta, net share issues, variance, and
    residual variance). Returns a wide DataFrame (columns = candidate
    factors) on the common month-end index.

    All spreads are Ken French decile high-minus-low value-weighted
    portfolios (or the published factor where available), so the universe
    is larger and more heterogeneous than the original 9-factor panel:
    it includes anomaly families (accruals, net issuance, low-volatility,
    low-beta) that the FF5 model does NOT span.
    """
    ff = get_ff_factor_frets()
    out = {"mkt": ff["MKT"].dropna()}
    for k in ("SMB", "HML", "RMW", "CMA"):
        out[k.lower()] = ff[k].dropna()
    out["mom"] = get_ff_momentum().dropna()

    # Momentum / reversal decile spreads (published decile files).
    spread_defs = {
        "umd_ls":   "10_Portfolios_Prior_12_2",
        "srev_ls":  "10_Portfolios_Prior_1_0",
        "lrev_ls":  "10_Portfolios_Prior_60_13",
    }
    for label, fn in spread_defs.items():
        try:
            out[label] = get_kf_decile_spread(fn, 10)
        except Exception as e:   # noqa: F841
            print(f"[warn] skipped {label}: {e}")

    # "Portfolios Formed on X" decile files -> characteristic H-L spreads.
    char_defs = {
        "me":      "Portfolios_Formed_on_ME",
        "bm":      "Portfolios_Formed_on_BE-ME",
        "ep":      "Portfolios_Formed_on_E-P",
        "cfp":     "Portfolios_Formed_on_CF-P",
        "dp":      "Portfolios_Formed_on_D-P",
        "op":      "Portfolios_Formed_on_OP",
        "inv":     "Portfolios_Formed_on_INV",
        "ac":      "Portfolios_Formed_on_AC",
        "beta":    "Portfolios_Formed_on_BETA",
        "ni":      "Portfolios_Formed_on_NI",
        "var":     "Portfolios_Formed_on_VAR",
        "resvar":  "Portfolios_Formed_on_RESVAR",
    }
    for label, fn in char_defs.items():
        try:
            out[label] = get_kf_char_decile_spread(fn)
        except Exception as e:   # noqa: F841
            print(f"[warn] skipped {label}: {e}")

    df = pd.DataFrame(out)
    df = df[df.index >= "1963-07-01"]
    # keep factors with enough history
    df = df[[c for c in df.columns if df[c].notna().sum() >= min_obs]]
    return df.sort_index()
