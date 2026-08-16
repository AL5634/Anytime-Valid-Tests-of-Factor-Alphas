"""
factor_metadata.py — canonical publication-year metadata for candidate factors.

Used to split each factor's e-process path into pre- and post-publication
segments for the factor-life-cycle (decay) diagnostic in the paper
(Section 5.2). Publication years are the canonical first public disclosure
of each anomaly in the academic literature; the first month of the
publication year is treated as the start of the "post" window.
"""
from __future__ import annotations

# factor column name -> (publication year, canonical source)
FACTOR_PUBLICATION_YEARS: dict[str, tuple[int, str]] = {
    # Momentum / reversal
    "mom":     (1993, "Jegadeesh and Titman (1993); Carhart (1997)"),
    "umd_ls":  (1993, "Jegadeesh and Titman (1993); Carhart (1997)"),
    "srev_ls": (1990, "Jegadeesh (1990); Lehmann (1990)"),
    "lrev_ls": (1985, "De Bondt and Thaler (1985)"),
    # Valuation / profitability / investment
    "bm":      (1993, "Fama and French (1993)"),
    "ep":      (1977, "Basu (1977)"),
    "cfp":     (1994, "Lakonishok, Shleifer and Vishny (1994)"),
    "dp":      (1988, "Fama and French (1988); Litzenberger and Ramaswamy (1979)"),
    "op":      (2013, "Novy-Marx (2013)"),
    "inv":     (2004, "Titman, Wei and Xie (2004)"),
    # Distress / earnings-quality / issuance
    "ac":      (1996, "Sloan (1996)"),
    "ni":      (2008, "Pontiff and Woodgate (2008)"),
    # Risk-based anomalies
    "beta":    (1972, "Black (1972); Black, Jensen and Scholes (1972)"),
    "var":     (2006, "Ang, Hodrick, Xing and Zhang (2006)"),
    "resvar":  (2006, "Ang, Hodrick, Xing and Zhang (2006)"),
    # Size
    "me":      (1981, "Banz (1981); Reinganum (1981)"),
    # Legacy 2x3 spreads (pre-expansion panel; kept for backward
    # compatibility with older result files)
    "op_ls":   (2013, "Novy-Marx (2013)"),
    "inv_ls":  (2004, "Titman, Wei and Xie (2004)"),
    "ep_ls":   (1977, "Basu (1977)"),
    "dp_ls":   (1988, "Fama and French (1988)"),
    "cfp_ls":  (1994, "Lakonishok, Shleifer and Vishny (1994)"),
}


def publication_date(factor: str) -> str | None:
    """Month-end publication date for a factor, or None if unknown.

    Returns the first month-end of the publication year (e.g. '1993-01-31'),
    which matches the month-end index of the e-process paths. With the
    `<=`/`>=` split used in `factor_survival_stats`, this places all months
    of the publication year in the "post" window.
    """
    year = FACTOR_PUBLICATION_YEARS.get(factor, (None, None))[0]
    if year is None:
        return None
    return f"{year}-01-31"


def all_publication_dates() -> dict[str, str]:
    """Map every known factor column name to its publication date string."""
    return {k: publication_date(k) for k in FACTOR_PUBLICATION_YEARS}
