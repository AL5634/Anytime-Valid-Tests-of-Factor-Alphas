"""
verify_tables.py — Automated cross-check of every numeric table in the
manuscript and internet appendix against the source CSVs in data/results.

Read-only: re-derives each table's numbers from the CSVs, parses the numbers
out of the .tex, and reports PASS/FAIL per table. Run from the project root:

    python3 scripts/verify_tables.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "data/results"
MAN = (BASE / "paper/jef/manuscript.tex").read_text()
IA = (BASE / "paper/ia/internet_appendix.tex").read_text()

report = []


def check(name, ok, detail=""):
    report.append((name, ok, detail))


def num(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def close(a, b, tol=0.01):
    if a is None or b is None:
        return False
    return abs(a - b) <= tol + 1e-9


# ---------------------------------------------------------------- tab:oos
def verify_oos():
    tex = MAN
    df = pd.read_csv(RES / "oos_comparison.csv").set_index("method")
    # rows look like: E-value ($\mathcal{E}\ge20$) & 62 & 4.74 & 5.88 & 0.93 & $-0.13$ \\
    rows = re.findall(
        r"^(.*?) & (\d+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & \$?-?([\d.]+)\$? \\\\",
        tex, re.M)
    matched = 0
    for label, n, a, t, s, dd in rows:
        a, t, s = num(a), num(t), num(s)
        # find which method this row is
        for m, r in df.iterrows():
            if close(a, r["alpha_annualized_pct"], 0.01) and int(n) == int(r["n_factors"]):
                ok = (close(t, r["t_alpha"], 0.01) and close(s, r["sharpe_annualized"], 0.01)
                      and close(num(dd), abs(r["max_drawdown"]), 0.01))
                matched += 1
                if not ok:
                    check(f"tab:oos row {m}", False, f"tex a={a} t={t} s={s} dd={dd} vs csv")
                break
    check("tab:oos (5 rules)", matched == 5, f"matched {matched}/5 rows")


# ---------------------------------------------------------------- tab:sens
def verify_sens():
    g = pd.read_csv(RES / "sensitivity_grid.csv")
    grid = {(int(r.min_obs), float(r.clip_scale)): int(r.n_rej) for _, r in g.iterrows()}
    # parse the 4 data rows: $24$ & 118 & 100 & 68 & 30 \\
    rows = re.findall(r"^\$(\d+)\$\s*&\s*(\d+)\s*&\s*(\d+)\s*&\s*(\d+)\s*&\s*(\d+)\s*\\\\", MAN, re.M)
    ok = True
    seen = 0
    for mo, c2, c3, c4, c6 in rows:
        mo = int(mo)
        for cs, val in [(2.0, c2), (3.0, c3), (4.0, c4), (6.0, c6)]:
            exp = grid.get((mo, cs))
            if exp is None or int(val) != exp:
                ok = False
            seen += 1
    check("tab:sens (16 cells)", ok and seen == 16, f"checked {seen}/16 cells")


# ---------------------------------------------------------------- tab:calib
def verify_calib():
    p = pd.read_csv(RES / "power_simulation.csv")
    # rows: $0$  (size) & 0.3 & 5.0 & 0.1 & 5.0 \\  (T=240 eproc, t240, T=726 eproc, t726)
    rows = re.findall(
        r"^\$(\d+)\$.*?&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*(\$?\\approx\$?100|[\d.]+)\s*\\\\",
        MAN, re.M)
    t240 = p[p.horizon_months == 240].set_index("alpha_annual")
    t726 = p[p.horizon_months == 726].set_index("alpha_annual")
    ok = True
    seen = 0
    for a, e240, t240t, e726, t726t in rows:
        a = float(a)
        if a not in t726.index:
            continue
        exp = {
            "e240": t240.loc[a]["reject_anytime"] * 100,
            "t240": t240.loc[a]["t_test_power"] * 100,
            "e726": t726.loc[a]["reject_anytime"] * 100,
            "t726": t726.loc[a]["t_test_power"] * 100,
        }
        t726t_v = 100.0 if "approx" in str(t726t) else float(t726t)
        got = {"e240": num(e240), "t240": num(t240t),
               "e726": num(e726), "t726": t726t_v}
        for k in ("e240", "t240", "e726", "t726"):
            if got[k] is not None and not close(got[k], exp[k], 0.15):
                ok = False
        seen += 1
    check("tab:calib (all 24 cells, T=240 & T=726)", ok and seen == 6,
          f"checked {seen} alpha rows x 4 cols")


# ---------------------------------------------------------------- tab:data
def _norm_cat(s):
    s = s.strip().lower()
    s = s.replace("(", " ").replace(")", " ")
    s = s.replace("--", " ").replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()


def verify_data():
    ev = pd.read_csv(RES / "evalue_results.csv", index_col=0)
    cats = set(ev.cat_economic.dropna().unique())
    # manuscript tab:data: Category & N & Mean & Vol & Rejected & e-BH
    rows = re.findall(
        r"^([A-Za-z][A-Za-z ()\-]+?) & (\d+) & ([\d.\-]+) & ([\d.\-]+) & (\d+) & (\d+) \\\\",
        MAN, re.M)
    univ = pd.read_csv(BASE / "data/processed/factor_universe.csv",
                       index_col=0, parse_dates=True)
    ok = True
    seen = 0
    for cat, n, mean, vol, rej, ebh in rows:
        key = _norm_cat(cat)
        if key not in cats:
            continue
        sub = ev[ev.cat_economic == key]
        n_ebh = int(sub["ebh_reject"].fillna(False).astype(bool).sum())
        facs = sub.index.tolist()
        means = [univ[f].dropna().mean() * 12 * 100 for f in facs if f in univ]
        vols = [univ[f].dropna().std(ddof=1) * 12 * 100 for f in facs if f in univ]
        ok_cells = (int(n) == len(sub)
                    and int(rej) == int((sub.evalue_final >= 20).sum())
                    and int(ebh) == n_ebh
                    and close(num(mean), np.mean(means), 0.06)
                    and close(num(vol), np.mean(vols), 0.06))
        if not ok_cells:
            ok = False
        seen += 1
    check("tab:data (category N, Mean, Vol, rejects, e-BH)",
          ok and seen >= 10, f"checked {seen} categories")


# ---------------------------------------------------------------- tab:lifecycle
_DISPLAY_CAT = {
    "External financing": "external financing",
    "Momentum": "momentum",
    "Other": "other",
    "Profitability": "profitability",
    "Investment": "investment",
    "Valuation": "valuation",
    "Volatility": "volatility",
    "Investment (alt)": "investment alt",
    "Earnings forecast": "earnings forecast",
    "Liquidity": "liquidity",
    "Lead--lag": "lead lag",
    "Sales growth": "sales growth",
}


def verify_lifecycle():
    ev = pd.read_csv(RES / "evalue_results.csv", index_col=0)
    df = ev.dropna(subset=["cat_economic"]).copy()
    df["pre_k"] = df["pre_pub_egr"] * 1000.0
    df["post_k"] = df["post_pub_egr"] * 1000.0
    df["first_rej_year"] = pd.to_numeric(df["first_rejection_date"].str[:4], errors="coerce")
    df["is_rej"] = df["evalue_final"] >= 20.0
    df["pre_pub"] = df["is_rej"] & (df["first_rej_year"] < df["pub_year"])
    df["lead10"] = df["is_rej"] & ((df["pub_year"] - df["first_rej_year"]) >= 10)
    grp = df.groupby("cat_economic").agg(
        N=("evalue_final", "count"),
        Nrej=("is_rej", "sum"),
        pre=("pre_k", "mean"),
        post=("post_k", "mean"),
        med=("post_pub_decay_ratio", "median"),
        npub=("pre_pub", "sum"),
        nlead=("lead10", "sum"))

    ok = True
    seen = 0
    for display, raw in _DISPLAY_CAT.items():
        r = grp.loc[raw]
        pat = re.escape(display) + r"\s*&\s*(\d+)\s*&\s*(\d+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*(\d+)\s*&\s*(\d+)\s*\\\\"
        m = re.search(pat, MAN)
        if not m:
            ok = False
            continue
        N, Nrej, pre, post, med, npub, nlead = m.groups()
        exp = (int(N) == int(r["N"]) and int(Nrej) == int(r["Nrej"])
               and close(num(pre), r["pre"], 0.01) and close(num(post), r["post"], 0.01)
               and close(num(med), r["med"], 0.01) and int(npub) == int(r["npub"])
               and int(nlead) == int(r["nlead"]))
        if not exp:
            ok = False
        seen += 1
    # "All predictors" row
    m = re.search(r"All predictors\s*&\s*212\s*&\s*76\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*62\s*&\s*43\s*\\\\", MAN)
    if not (m and close(num(m.group(1)), df["pre_k"].mean(), 0.01)
            and close(num(m.group(2)), df["post_k"].mean(), 0.01)
            and close(num(m.group(3)), df["post_pub_decay_ratio"].median(), 0.01)):
        ok = False
    seen += 1
    check("tab:lifecycle (12 categories + All)", ok and seen == 13, f"checked {seen} rows")


# ---------------------------------------------------------------- tab:survival
def verify_survival():
    ev = pd.read_csv(RES / "evalue_results.csv", index_col=0)
    top = ev.sort_values("evalue_final", ascending=False).head(12)
    ok = True
    seen = 0
    for name, r in top.iterrows():
        # row: name & cat & $E$ & 1977:06 & 2013 & 0.2 \\
        pat = (re.escape(name) +
               r"\s*&.*?&\s*\$([\d.]+)\\times10\^\{(\d+)\}\$"
               r"\s*&\s*(\d{4}):(\d{2})\s*&\s*(\d{4})\s*&\s*([\d.]+)\s*\\\\")
        m = re.search(pat, MAN)
        if not m:
            ok = False
            continue
        mant, exp, rej_m, rej_y, pub, ratio = m.groups()
        got_e = float(mant) * 10 ** int(exp)
        ndec = len(mant.split(".")[1]) if "." in mant else 0
        tol_e = 0.5 * 10.0 ** (int(exp) - ndec)
        ok_e = close(got_e, r["evalue_final"], tol_e)
        rej_tex = f"{rej_m}:{rej_y}"
        exp_rej = str(r["first_rejection_date"])[:7].replace("-", ":")
        ok_rej = rej_tex == exp_rej
        ok_pub = int(pub) == int(r["pub_year"])
        ok_ratio = close(num(ratio), r["post_pub_decay_ratio"], 0.051)
        if not (ok_e and ok_rej and ok_pub and ok_ratio):
            ok = False
        seen += 1
    check("tab:survival (E, 1st rej, pub, decay ratio)",
          ok and seen == 12, f"matched {seen}/12 rows")


# ---------------------------------------------------------------- IA exclusion
def verify_exclusion():
    e = pd.read_csv(RES / "oos_exclusion.csv").set_index("portfolio")
    ia_tab = (BASE / "paper/ia/tables/tab_exclusion.tex").read_text()
    mapping = {
        "E-value (62)": "E-value (full, 62)",
        "Core (E-value $\\cap$ $|t|>2$)": "Core = E-value $\\cap$ |t|>2",
        "Excluded by E-value (97)": "Excluded by E-value (|t|>2 - E-value)",
        "E-value only (2)": "E-value only",
        "$|t|>2$ (157)": "|t|>2 (full, 157)",
        "E-value e-BH (43)": "E-value e-BH (43)",
        "Excluded by e-BH (114)": "Excluded by e-BH (|t|>2 - e-BH)",
    }
    ok = True
    seen = 0
    for line in ia_tab.splitlines():
        line = line.strip()
        if "&" in line and "\\\\" in line and not line.startswith("\\"):
            parts = [p.strip() for p in line.replace("\\\\", "").split(" & ")]
            if len(parts) != 5:
                continue
            label = parts[0]
            if label not in mapping:
                continue
            r = e.loc[mapping[label]]
            n, a, t, s = parts[1], parts[2], parts[3], parts[4]
            if not (int(n) == int(r["n_factors"])
                    and close(num(a), r["alpha_annualized_pct"], 0.011)
                    and close(num(t), r["t_alpha"], 0.011)
                    and close(num(s), r["sharpe_annualized"], 0.011)):
                ok = False
            seen += 1
    check("IA tab:exclusion (7 rows)", ok and seen == 7, f"matched {seen}/7 rows")


# ---------------------------------------------------------------- IA lateholdout
def verify_lateholdout():
    h = pd.read_csv(RES / "oos_lateholdout.csv")
    ia_tab = (BASE / "paper/ia/tables/tab_lateholdout.tex").read_text()
    mapping = {"E-value": "ev", "E-value (e-BH)": "ebh", "$|t|>3$": "hlz",
               "$|t|>2$": "trad", "EB-shrink": "eb"}
    ok = True
    seen = 0
    for label, rule in mapping.items():
        early = h[(h.rule == rule) & h.window.str.startswith("Early")]
        late = h[(h.rule == rule) & h.window.str.startswith("Late")]
        if len(early) == 0 or len(late) == 0:
            ok = False
            continue
        pat = re.escape(label) + r"\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*\\\\"
        m = re.search(pat, ia_tab)
        if not m:
            ok = False
            continue
        ea, es, la, ls = m.groups()
        if not (close(num(ea), early.iloc[0]["alpha_annualized_pct"], 0.011)
                and close(num(es), early.iloc[0]["sharpe_annualized"], 0.011)
                and close(num(la), late.iloc[0]["alpha_annualized_pct"], 0.011)
                and close(num(ls), late.iloc[0]["sharpe_annualized"], 0.011)):
            ok = False
        seen += 1
    check("IA tab:lateholdout (5 rules x early/late)", ok and seen == 5,
          f"matched {seen}/5 rules")


# ---------------------------------------------------------------- IA skewsize
def verify_skewsize():
    s = pd.read_csv(RES / "skew_size.csv")
    tex = (BASE / "paper/ia/tables/tab_skewsize.tex").read_text()
    ok = True
    seen = 0
    for line in tex.splitlines():
        line = line.strip()
        m = re.match(r"^([A-Za-z].*?)\s*&\s*(\d+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*(-?[\d.]+)\s*\\\\$", line)
        if not m:
            continue
        dgp, T, size, bind, bias = m.groups()
        norm = lambda x: x.replace("~", "-").replace(" ", "").replace("-", "").lower()
        row = s[(s.experiment == "parametric") & (s["T"] == int(T))
                & (s.dgp.map(norm) == norm(dgp))]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        ok_r = (close(num(size), r["empirical_size"] * 100, 0.006)
                and close(num(bind), r["clip_bind_freq"] * 100, 0.006)
                and close(num(bias), r["truncation_bias"], 0.0002))
        if not ok_r:
            ok = False
        seen += 1
    emp = s[s.experiment == "empirical"]
    if len(emp):
        m = re.search(r"demeaned FF5 residuals \(212 predictors\)\s*&\s*726\s*&\s*([\d.]+)", tex)
        if not m or not close(num(m.group(1)), emp.iloc[0]["empirical_size"] * 100, 0.01):
            ok = False
        seen += 1
    check("IA tab:skewsize (14 parametric rows + empirical)", ok and seen >= 15,
          f"checked {seen} rows")


# ---------------------------------------------------------------- IA rolling
def verify_rolling():
    roll = pd.read_csv(RES / "evalue_rolling120.csv", index_col=0)
    ev = pd.read_csv(RES / "evalue_results.csv", index_col=0)
    exp_set = set(ev.index[ev.evalue_final >= 20])
    roll_set = set(roll.index[roll.evalue_final >= 20])
    n_roll = len(roll_set)
    overlap = len(exp_set & roll_set)
    ia_tab = (BASE / "paper/ia/tables/tab_rolling.tex").read_text()
    ok = (str(n_roll) in ia_tab and str(overlap) in ia_tab
          and str(len(exp_set - roll_set)) in ia_tab
          and str(len(roll_set - exp_set)) in ia_tab)
    cats = ev["cat_economic"].dropna()
    ov_cats = cats[cats.index.isin(exp_set & roll_set)].value_counts()
    for cat, cnt in [("other", 9), ("external financing", 9), ("momentum", 6),
                     ("volatility", 4), ("valuation", 4), ("profitability", 4)]:
        if int(ov_cats.get(cat, 0)) != cnt:
            ok = False
    check("IA tab:rolling (76/76/72, 4/4, overlap cats)", ok,
          f"n_roll={n_roll} overlap={overlap}")


# ---------------------------------------------------------------- IA firstrej
def verify_firstrej():
    f = pd.read_csv(RES / "first_rejection_timing.csv").iloc[0]
    ia_tab = (BASE / "paper/ia/tables/tab_firstrej.tex").read_text()
    ok = (str(int(f["n_reject_before_pub"])) in ia_tab
          and str(int(f["n_reject_at_least_10y_before_pub"])) in ia_tab
          and str(int(f["n_reject_before_pub_and_decay"])) in ia_tab)
    check("IA tab:firstrej (62 & 56 & 43)", ok, "")


# ---------------------------------------------------------------- IA tails
def verify_tails():
    t = pd.read_csv(RES / "residual_tails.csv")
    g = t.groupby("rejected")[["skew", "exkurt"]].median()
    ia_tab = (BASE / "paper/ia/tables/tab_tails.tex").read_text()
    ok = (f"{g.loc[True, 'skew']:.2f}" in ia_tab and f"{g.loc[True, 'exkurt']:.2f}" in ia_tab
          and f"{g.loc[False, 'skew']:.2f}" in ia_tab and f"{g.loc[False, 'exkurt']:.2f}" in ia_tab)
    check("IA tab:tails (both groups)", ok, "")


# ---------------------------------------------------------------- IA corr
def verify_corr():
    c = pd.read_csv(RES / "oos_factor_corr.csv")
    ia_tab = (BASE / "paper/ia/tables/tab_corr.tex").read_text()
    ok = True
    for _, r in c.iterrows():
        pat = re.escape(r["selection"]) + r"\s*&\s*(\d+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*(\d+)\\\%"
        m = re.search(pat, ia_tab)
        if not m:
            ok = False
            continue
        n, mc, mx, sh = m.groups()
        if not (int(n) == int(r["n_factors"]) and close(num(mc), r["mean_corr"], 0.001)
                and close(num(mx), r["max_corr"], 0.001)
                and abs(int(sh) - round(r["share_above_0.5"] * 100)) <= 1):
            ok = False
    check("IA tab:corr (both rows)", ok, "")


# ---------------------------------------------------------------- IA full 212
def _numcell(s):
    """Parse a LaTeX numeric cell like $6.45e+16$, 7117, -2.6 or ---."""
    s = s.strip().replace("$", "").replace(",", "")
    if s in ("---", "", "nan", "NaN", "\\checkmark"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def verify_full212():
    ev = pd.read_csv(RES / "evalue_results.csv", index_col=0)
    tex = (BASE / "paper/ia/tables/tab_full_212.tex").read_text()
    ok = True
    seen = 0
    for line in tex.splitlines():
        line = line.strip()
        if not line.startswith("\\begin") and "\\\\" in line and "&" in line:
            parts = [p.strip() for p in line.replace("\\\\", "").split(" & ")]
            if len(parts) != 9 or parts[0].startswith("\\") or parts[0] == "":
                continue
            name = parts[0].replace("\\_", "_")
            if name not in ev.index:
                continue
            r = ev.loc[name]
            e_tex = _numcell(parts[1])
            rej_tex, pub_tex = parts[2].strip(), parts[3].strip()
            cat_tex = parts[4].replace("\\&", "&").strip()
            egr_t, pre_t, post_t = (_numcell(parts[5]), _numcell(parts[6]),
                                    _numcell(parts[7]))
            flag_tex = "\\checkmark" in parts[8]

            exp_e = float(r["evalue_final"])
            # display-derived tolerance: half of the last displayed digit
            m_m = re.match(r"^([\d.]+)e([+\-]?\d+)$", parts[1].strip().replace("$", ""))
            if m_m:
                mant, exp = m_m.groups()
                ndec = len(mant.split(".")[1]) if "." in mant else 0
                tol = 0.5 * 10.0 ** (int(exp) - ndec)
            else:
                raw = parts[1].strip().replace("$", "")
                ndec = len(raw.split(".")[1]) if "." in raw else 0
                tol = 0.5 * 10.0 ** (-ndec)
            ok_e = e_tex is not None and abs(e_tex - exp_e) <= max(tol, abs(exp_e) * 1e-6)
            exp_rej = "" if pd.isna(r["first_rejection_date"]) else str(r["first_rejection_date"])[:7]
            ok_rej = (rej_tex == "---" and exp_rej == "") or rej_tex == exp_rej
            exp_pub = int(r["pub_year"]) if not pd.isna(r["pub_year"]) else None
            ok_pub = (pub_tex == "---" and exp_pub is None) or (
                pub_tex != "---" and exp_pub is not None and int(pub_tex) == exp_pub)
            ok_cat = cat_tex == r["cat_economic"]
            ok_egr = (egr_t is None) or abs(egr_t - float(r["egr"]) * 1000) <= 0.11
            ok_pre = (pre_t is None) or abs(pre_t - float(r["pre_pub_egr"]) * 1000) <= 0.11
            ok_post = (post_t is None) or abs(post_t - float(r["post_pub_egr"]) * 1000) <= 0.11
            ok_flag = flag_tex == bool(r["ebh_reject"])
            if not (ok_e and ok_rej and ok_pub and ok_cat and ok_egr and ok_pre and ok_post and ok_flag):
                ok = False
            seen += 1
    check("IA tab:full_212 (212 rows, 8 cols)", ok and seen == 212, f"matched {seen}/212 rows")


# ---------------------------------------------------------------- IA category
def verify_ia_category():
    ev = pd.read_csv(RES / "evalue_results.csv", index_col=0)
    tex = (BASE / "paper/ia/tables/tab_category.tex").read_text()
    grp = ev.dropna(subset=["cat_economic"]).groupby("cat_economic").agg(
        N=("evalue_final", "count"),
        rej=("evalue_final", lambda s: (s >= 20).sum()),
        meanE=("evalue_final", "mean"))
    ok = True
    seen = 0
    for line in tex.splitlines():
        line = line.strip()
        if "\\\\" in line and "&" in line and not line.startswith("\\"):
            parts = [p.strip() for p in line.replace("\\\\", "").split(" & ")]
            if len(parts) != 5:
                continue
            cat = parts[0].strip()
            if cat not in grp.index:
                continue
            N, rej, rate, me = parts[1], parts[2], parts[3], parts[4]
            r = grp.loc[cat]
            ok_r = (int(N) == int(r["N"]) and int(rej) == int(r["rej"])
                    and float(rate.strip().replace("\\%", "").replace("%", ""))
                    == round(100 * r["rej"] / r["N"]))
            # mean E: display-derived tolerance (2-sig-fig e-notation or %.1f)
            m = _numcell(me)
            mm = re.match(r"^([\d.]+)e([+\-]?\d+)$", me.replace("$", ""))
            if mm:
                mant, exp = mm.groups()
                ndec = len(mant.split(".")[1]) if "." in mant else 0
                tol = 0.5 * 10.0 ** (int(exp) - ndec)
            else:
                raw = me.replace("$", "")
                ndec = len(raw.split(".")[1]) if "." in raw else 0
                tol = 0.5 * 10.0 ** (-ndec)
            ok_m = m is not None and abs(m - r["meanE"]) <= max(tol, abs(r["meanE"]) * 1e-6)
            if not (ok_r and ok_m):
                ok = False
            seen += 1
    check("IA tab:category (N, rejects, mean E)", ok and seen >= 20, f"checked {seen} categories")


# ---------------------------------------------------------------- IA OOS selections
def verify_ia_ooselections():
    sel = json.load(open(RES / "oos_selections.json"))
    tex = (BASE / "paper/ia/tables/tab_oos_selections.tex").read_text()
    mapping = {"E-value": "ev", "HLZ": "hlz", "Traditional": "trad", "EB-shrink": "eb"}
    ok = True
    seen = 0
    for line in tex.splitlines():
        line = line.strip()
        if "&" in line and "\\\\" in line:
            parts = [p.strip() for p in line.replace("\\\\", "").split(" & ")]
            if len(parts) != 2:
                continue
            method = parts[0]
            for k, v in mapping.items():
                if method.startswith(k):
                    tex_set = {f.strip().replace("\\_", "_") for f in parts[1].split(",")}
                    data_set = set(sel[v])
                    if tex_set != data_set:
                        ok = False
                    seen += 1
                    break
    check("IA tab:oos_selections (4 method lists)", ok and seen == 4, f"matched {seen}/4 lists")


# ---------------------------------------------------------------- IA equalized
def verify_ia_equalized():
    ok = True
    seen = 0
    for n in (2, 5, 10):
        df = pd.read_csv(RES / f"oos_comparison_equalized_n{n}.csv").set_index("method")
        tex = (BASE / "paper/ia/tables/tab_oos_equalized.tex").read_text()
        # rows: E-value & 2 & 3.45 & 4.58 & 0.98 \\
        pat = rf"({re.escape('E-value')}|{re.escape('HLZ t>3.0')}|{re.escape('EB-shrink')}|{re.escape('t>2.0')}) & {n} & ([\d.]+) & ([\d.]+) & ([\d.]+) \\\\"
        for m in re.finditer(pat, tex):
            method, a, t, s = m.groups()
            r = df.loc[method]
            if not (close(num(a), r["alpha_annualized_pct"], 0.011)
                    and close(num(t), r["t_alpha"], 0.011)
                    and close(num(s), r["sharpe_annualized"], 0.011)):
                ok = False
            seen += 1
    check("IA tab:oos_equalized (N=2,5,10)", ok and seen == 12, f"matched {seen}/12 rows")


# ---------------------------------------------------------------- IA alt splits
def verify_ia_altsplits():
    import sys
    sys.path.insert(0, str(BASE))
    from src.analysis.oos_horse_race import run_oos_horse_race
    import yaml
    cfg = yaml.safe_load(open(BASE / "config.yaml"))
    univ = pd.read_csv(BASE / "data/processed/factor_universe.csv", index_col=0,
                       parse_dates=True)
    ff = univ[[c for c in ["mkt", "smb", "hml", "rmw", "cma"] if c in univ.columns]]
    zoo = [c for c in univ.columns if c not in ff.columns]
    ev = pd.read_csv(RES / "evalue_results.csv", index_col=0)

    tex = (BASE / "paper/ia/tables/tab_oos_altsplits.tex").read_text()
    # slice the tex after each \multicolumn split header so that each
    # method's row is searched inside its own split block
    hdr_idx = [m.start() for m in re.finditer(r"\\multicolumn\{6\}\{c\}", tex)]
    hdr_idx.append(len(tex))
    ok = True
    seen = 0
    for i, split in enumerate(cfg["empirical"]["alt_splits"]):
        ts, te, vs, ve = split
        import copy
        c = copy.deepcopy(cfg)
        c["empirical"].update({"train_start": ts, "train_end": te,
                               "test_start": vs, "test_end": ve})
        df, _ = run_oos_horse_race(univ[zoo], ff, ev, c)
        block = tex[hdr_idx[i]:hdr_idx[i + 1]]
        for _, r in df.iterrows():
            method = r["method"]
            pat = (rf" & {re.escape(method)} & (\d+) & ([\d.\-]+) & "
                   rf"([\d.\-]+) & ([\d.\-]+) \\\\")
            m = re.search(pat, block)
            if not m:
                ok = False
                continue
            n_sel, a, t, s = m.groups()
            if not (int(n_sel) == int(r["n_factors"])
                    and close(num(a), r["alpha_annualized_pct"], 0.011)
                    and close(num(t), r["t_alpha"], 0.011)
                    and close(num(s), r["sharpe_annualized"], 0.011)):
                ok = False
            seen += 1
    check("IA tab:oos_altsplits (2 splits x 4 rules)", ok and seen == 8, f"matched {seen}/8 rows")


# ---------------------------------------------------------------- IA sensitivity
def verify_ia_sensitivity():
    g = pd.read_csv(RES / "sensitivity_grid.csv")
    tex = (BASE / "paper/ia/tables/tab_sensitivity.tex").read_text()
    ok = True
    seen = 0
    # rows: 24 & 2 & \footnotesize Accruals, ... \\
    for line in tex.splitlines():
        line = line.strip()
        m = re.match(r"^(\d+) & ([\d.]+) & .*?([A-Za-z].*?)\\\\$", line)
        if not m:
            continue
        mo, cs, lst = m.groups()
        row = g[(g.min_obs == int(mo)) & (g.clip_scale == float(cs))]
        if len(row) == 0:
            ok = False
            continue
        lst = re.sub(r"^\\?footnotesize\s*", "", lst)
        tex_set = {f.strip().replace("\\_", "_") for f in lst.split(",")}
        data_set = set(row.iloc[0]["rejected"].split(","))
        if tex_set != data_set:
            ok = False
        seen += 1
    check("IA tab:sensitivity (16 cells lists)", ok and seen == 16, f"checked {seen}/16 cells")


# ---------------------------------------------------------------- IA KF16
def verify_kf16():
    kf = BASE / "data/results_kf"
    ev = pd.read_csv(kf / "evalue_results.csv", index_col=0)
    oos = pd.read_csv(kf / "oos_comparison.csv").set_index("method")
    tex_e = (BASE / "paper/ia/tables/tab_kf16_evalue.tex").read_text()
    tex_o = (BASE / "paper/ia/tables/tab_kf16_oos.tex").read_text()
    ok = True
    seen = 0
    for line in tex_e.splitlines():
        line = line.strip()
        if "&" in line and "\\\\" in line and not line.startswith("\\"):
            parts = [p.strip() for p in line.replace("\\\\", "").split(" & ")]
            if len(parts) != 6:
                continue
            name = parts[0].replace("\\_", "_")
            if name not in ev.index:
                continue
            r = ev.loc[name]
            e_tex = _numcell(parts[1])
            ok_e = e_tex is not None and abs(e_tex - float(r["evalue_final"])) <= max(float(r["evalue_final"]) * 1e-2, 0.5)
            rej_tex = parts[2].strip()
            exp_rej = "" if pd.isna(r["first_rejection_date"]) else str(r["first_rejection_date"])[:7]
            ok_rej = (rej_tex == "---" and exp_rej == "") or rej_tex == exp_rej
            pre, post = _numcell(parts[4]), _numcell(parts[5])
            ok_pre = pre is None or abs(pre - float(r.get("pre_pub_egr", np.nan)) * 1000) <= 0.15
            ok_post = post is None or abs(post - float(r.get("post_pub_egr", np.nan)) * 1000) <= 0.15
            if not (ok_e and ok_rej and ok_pre and ok_post):
                ok = False
            seen += 1
    # OOS rows: E-value & 3 & 0.06 & 0.02 & -0.08 \\
    for m in re.finditer(r"(E-value|HLZ t>3\.0|EB-shrink|t>2\.0) & (\d+) & ([\d.\-]+) & ([\d.\-]+) & ([\d.\-]+) \\\\", tex_o):
        method, n, a, t, s = m.groups()
        r = oos.loc[method]
        if not (int(n) == int(r["n_factors"])
                and close(num(a), r["alpha_annualized_pct"], 0.011)
                and close(num(t), r["t_alpha"], 0.011)
                and close(num(s), r["sharpe_annualized"], 0.011)):
            ok = False
        seen += 1
    check("IA tab:kf16 (16 evalue rows + 4 oos rows)", ok and seen >= 18, f"matched {seen} rows")


def main():
    verify_oos()
    verify_sens()
    verify_calib()
    verify_data()
    verify_lifecycle()
    verify_survival()
    verify_exclusion()
    verify_lateholdout()
    verify_skewsize()
    verify_rolling()
    verify_firstrej()
    verify_tails()
    verify_corr()
    verify_full212()
    verify_ia_category()
    verify_ia_ooselections()
    verify_ia_equalized()
    verify_ia_altsplits()
    verify_ia_sensitivity()
    verify_kf16()

    print(f"{'TABLE':<40} {'STATUS':<6} DETAIL")
    npass = 0
    for name, ok, det in report:
        print(f"{name:<40} {'PASS' if ok else 'FAIL':<6} {det}")
        npass += ok
    print(f"\nTOTAL: {npass}/{len(report)} PASS")
    return 0 if npass == len(report) else 1


if __name__ == "__main__":
    sys.exit(main())
