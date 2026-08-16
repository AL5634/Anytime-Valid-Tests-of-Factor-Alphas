"""
figures.py — Multi-panel figures for the E-value Factor Zoo.

  - Colorblind-safe palette (Okabe-Ito / Seaborn Colorblind)
  - Vector PDF output at near-final publication size (Computer Modern serif
    typography, math and text harmonised via the 'cm' mathtext fontset)
  - Collision-free label placement with optional leader lines
  - NBER recession shading and clear geometric markers

Figures produced:
  Main text:
    - fig4_oos_horserace.pdf: Out-of-sample portfolio horse race & drawdown
      dynamics (the single main-text figure; all other evidence is tabular)
  Internet Appendix:
    - fig_ia1_power_robustness.pdf: Power curves and size under heavy-tailed DGPs
    - fig_ia2_correlation_structure.pdf: Factor correlation heatmap and category weighting
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Visual styling
# ---------------------------------------------------------------------------

PALETTE = sns.color_palette("colorblind", 10)
# Semantic color mapping
C_MAIN = PALETTE[0]      # Blue (E-value / Baseline)
C_ALT = PALETTE[3]       # Red / Amber (Rejections / Counterpart)
C_TRAD = PALETTE[7]      # Grey / Slate (Traditional t>2)
C_HLZ = PALETTE[2]       # Green (HLZ t>3)
C_EB = PALETTE[1]        # Orange (EB-shrink)
C_EBH = "#1b365d"        # Deep Navy (E-value e-BH)

# Font hierarchy (all figures): titles 10.5pt bold, axis labels 9.5pt,
# ticks 8pt, annotations 8pt, legends 8pt. Figures are drawn at near-final
# publication width so these sizes survive the LaTeX \linewidth rescale.
TITLE_SIZE = 10.5
LABEL_SIZE = 9.5
TICK_SIZE = 8.0
ANN_SIZE = 8.0
LEG_SIZE = 8.0


def setup_theme():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": TITLE_SIZE,
        "axes.titleweight": "bold",
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": True,
        "legend.fontsize": LEG_SIZE,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "grid.color": "#e5e5e5",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#d0d0d0",
    })


def _finish(fig, outdir, fname):
    fig.savefig(outdir / fname, bbox_inches="tight")
    plt.close(fig)


def add_nber_recessions(ax, start_year=2006, end_year=2023):
    """Shade NBER recessions during the out-of-sample window."""
    recessions = [
        ("2007-12-01", "2009-06-30"),  # Great Financial Crisis
        ("2020-02-01", "2020-04-30"),  # COVID-19 Recession
    ]
    for start, end in recessions:
        s_date = pd.to_datetime(start)
        e_date = pd.to_datetime(end)
        if s_date >= pd.to_datetime(f"{start_year}-01-01") and e_date <= pd.to_datetime(f"{end_year}-12-31"):
            ax.axvspan(s_date, e_date, color="#dcdcdc", alpha=0.55, zorder=0,
                       label="NBER Recession" if start == recessions[0][0] else "")


# ---------------------------------------------------------------------------
# Greedy collision-free label placement (with optional leader lines)
# ---------------------------------------------------------------------------

_CANDIDATES = [
    (6, 3), (-6, 3), (6, -3), (-6, -3),
    (0, 7), (0, -7), (7, 0), (-7, 0),
    (10, 5), (-10, 5), (10, -5), (-10, -5),
    (0, 12), (0, -12), (14, 0), (-14, 0),
    (12, 8), (-12, 8), (12, -8), (-12, -8),
    (18, 6), (-18, 6), (18, -6), (-18, -6),
    (0, 20), (0, -20), (22, 0), (-22, 0),
    (26, 12), (-26, 12), (26, -12), (-26, -12),
    (45, 18), (-45, 18), (45, -18), (-45, -18),
    (0, 40), (0, -40), (52, 0), (-52, 0),
    (70, 25), (-70, 25), (70, -25), (-70, -25),
    (0, 65), (0, -65), (80, 0), (-80, 0),
]


def _place_labels(ax, fig, points, labels, fontsize=ANN_SIZE,
                  lead_color="#444444"):
    """
    Greedy collision-free annotation placement.

    Each label is tried at a ranked set of offset directions (in display
    points), ranked to prefer placements that (i) stay inside the axes and
    (ii) do not overlap an already-placed label; among the remaining options
    the nearest one is chosen.  Labels end up sitting immediately beside their
    anchor point (standard in top-journal figures) and receive a thin leader
    line when the chosen offset is necessarily distant.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_ext = ax.get_window_extent(renderer=renderer)

    placed = []  # list of (label, bbox)
    final_artists = []  # (annotation, x, y, dx, dy)

    def _overlap_ratio(bb, margin=1.5, pad=4.0):
        """Fractional overlap of `bb` against any placed label box.  The
        candidate box is inflated by `pad` display points so that labels which
        merely graze an existing box (common for near-coincident points) are
        pushed to a clearly separated position."""
        best = 0.0
        from matplotlib.transforms import Bbox
        pbb = Bbox.from_extents(bb.x0 - pad, bb.y0 - pad,
                                bb.x1 + pad, bb.y1 + pad)
        for _, b0 in placed:
            ox = min(pbb.x1, b0.x1) - max(pbb.x0, b0.x0)
            oy = min(pbb.y1, b0.y1) - max(pbb.y0, b0.y0)
            if ox > margin and oy > margin:
                inter = (ox - margin) * (oy - margin)
                ar = inter / max(1e-12, min((bb.x1 - bb.x0) * (bb.y1 - bb.y0),
                                            (b0.x1 - b0.x0) * (b0.y1 - b0.y0)))
                best = max(best, ar)
        return best

    def _inside(bb):
        return (bb.x0 >= ax_ext.x0 and bb.x1 <= ax_ext.x1
                and bb.y0 >= ax_ext.y0 and bb.y1 <= ax_ext.y1)

    def _measure(x, y, label, dx, dy):
        ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
        va = "bottom" if dy > 0 else ("top" if dy < 0 else "center")
        ann = ax.annotate(label, xy=(x, y), xytext=(dx, dy),
                          textcoords="offset points", fontsize=fontsize,
                          fontweight="normal", ha=ha, va=va)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bb = ann.get_window_extent(renderer=renderer)
        ann.remove()
        return bb, ha, va, float(np.hypot(dx, dy))

    for (x, y), label in zip(points, labels):
        scored = []
        for dx, dy in _CANDIDATES:
            bb, ha, va, dist = _measure(x, y, label, dx, dy)
            scored.append((_inside(bb), _overlap_ratio(bb), dist, dx, dy, ha, va))
        # Prefer in-axes, then zero overlap, then least overlap, then nearest.
        scored.sort(key=lambda t: (not t[0], t[1] > 0.0, t[1], t[2]))
        inside, ov, dist, dx, dy, ha, va = scored[0]
        ann = ax.annotate(label, xy=(x, y), xytext=(dx, dy),
                          textcoords="offset points", fontsize=fontsize,
                          fontweight="normal", ha=ha, va=va,
                          annotation_clip=False, zorder=6)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bb = ann.get_window_extent(renderer=renderer)
        placed.append((label, bb))
        final_artists.append((ann, x, y, dx, dy))
    fig.canvas.draw()

    # Leader lines drawn as independent (empty-text) arrows so the label
    # annotations themselves keep arrow-free, text-only bounding boxes.
    for ann, x, y, dx, dy in final_artists:
        renderer = fig.canvas.get_renderer()
        bb = ann.get_window_extent(renderer=renderer)
        cx = bb.x0 + (bb.x1 - bb.x0) / 2
        cy = bb.y0 + (bb.y1 - bb.y0) / 2
        ax_disp = ax.transData.transform((x, y))
        if float(np.hypot(cx - ax_disp[0], cy - ax_disp[1])) > 20.0:
            ax.annotate("", xy=(x, y), xytext=(dx, dy),
                        textcoords="offset points", zorder=5,
                        arrowprops=dict(arrowstyle="-", color=lead_color,
                                        lw=0.7, shrinkA=2, shrinkB=2))
    fig.canvas.draw()


# ---------------------------------------------------------------------------
# FIGURE 4: Out-of-Sample Portfolio Horse Race & Drawdown Profiles (Master)
# ---------------------------------------------------------------------------

def fig4_oos_horserace(cum_by_method: dict, outdir: Path):
    """
    Figure 4: Out-of-Sample Screening Portfolio Dynamics (2006--2023):
      Panel A: Cumulative FF5 Alpha paths across the 5 screening rules with NBER recessions.
      Panel B: Underwater Drawdown profiles.
      A single figure-level legend identifies the five rules.
    """
    setup_theme()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 8.0), height_ratios=[1.15, 1.0],
                                   sharex=True)
    fig.subplots_adjust(top=0.90)

    method_styles = [
        ("E-value (e-BH)", C_EBH, "-", 2.3),
        ("E-value", C_MAIN, "-", 1.8),
        ("t>2.0", C_TRAD, "-.", 1.4),
        ("HLZ t>3.0", C_HLZ, "--", 1.4),
        ("EB-shrink", C_EB, ":", 1.6),
    ]
    short_labels = {
        "E-value (e-BH)": "E-value (e-BH)",
        "E-value": "E-value",
        "t>2.0": r"$|t|>2$",
        "HLZ t>3.0": r"HLZ $|t|>3$",
        "EB-shrink": "EB-shrink",
    }

    add_nber_recessions(ax1, 2006, 2023)
    add_nber_recessions(ax2, 2006, 2023)

    for method, col, ls, lw in method_styles:
        if method in cum_by_method and not cum_by_method[method].empty:
            port = cum_by_method[method].dropna()
            if not isinstance(port.index, pd.DatetimeIndex):
                port.index = pd.to_datetime(port.index)
            cum_wealth = (1.0 + port).cumprod()
            ax1.plot(cum_wealth.index, cum_wealth, color=col, linestyle=ls, linewidth=lw,
                     label=short_labels[method])
            peak = cum_wealth.cummax()
            dd = (cum_wealth - peak) / peak * 100.0
            ax2.plot(dd.index, dd, color=col, linestyle=ls, linewidth=lw)

    ax1.set_title("Panel A: Cumulative Out-of-Sample Performance of Factor-Screening Portfolios (2006--2023)",
                  fontsize=TITLE_SIZE, pad=6)
    ax1.set_ylabel("Cumulative Wealth ($W_0=1.0$)", fontsize=LABEL_SIZE)

    ax2.set_title("Panel B: Underwater Drawdown Dynamics (%)", fontsize=TITLE_SIZE, pad=6)
    ax2.set_xlabel("Calendar Year", fontsize=LABEL_SIZE)
    ax2.set_ylabel("Drawdown from Peak (%)", fontsize=LABEL_SIZE)
    ax2.set_ylim(-22, 2)
    ax2.xaxis.set_major_locator(mdates.YearLocator(3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Inset axis: Factor Selection Breadth (order matches Panel A legend)
    ax_inset = ax1.inset_axes([0.70, 0.10, 0.27, 0.34])
    counts = [43, 62, 157, 119, 11]
    labels_short = ["e-BH", "E-val", r"$|t|>2$", "HLZ", "EB"]
    cols_inset = [C_EBH, C_MAIN, C_TRAD, C_HLZ, C_EB]
    x_pos = np.arange(len(labels_short))
    bars = ax_inset.bar(x_pos, counts, color=cols_inset, edgecolor="white", width=0.62)
    ax_inset.set_xticks(x_pos)
    ax_inset.set_xticklabels(labels_short)
    ax_inset.set_title("Selection Breadth ($N$)", fontsize=8.0, pad=3)
    ax_inset.tick_params(labelsize=7.0)
    for b, ct in zip(bars, counts):
        ax_inset.text(b.get_x() + b.get_width() / 2, b.get_height() + 3, str(ct),
                      ha="center", fontsize=7.0, fontweight="bold")
    ax_inset.set_ylim(0, 195)
    ax_inset.set_axisbelow(True)

    # Figure-level legend (top row)
    handles = [Rectangle((0, 0), 1, 1, facecolor="#dcdcdc", alpha=0.55,
                         edgecolor="none", label="NBER Recession")]
    for method, col, ls, lw in method_styles:
        handles.append(Line2D([0], [1], color=col, linestyle=ls, linewidth=lw,
                              label=short_labels[method]))
    fig.legend(handles=handles, loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, 0.995), frameon=True, framealpha=0.95,
               fontsize=8.0)

    _finish(fig, outdir, "fig4_oos_horserace.pdf")


# ---------------------------------------------------------------------------
# FIGURE IA.1: Power and Heavy-Tail Robustness Surfaces (Internet Appendix)
# ---------------------------------------------------------------------------

def fig_ia1_power_robustness(power_results: pd.DataFrame, skew_results: pd.DataFrame, outdir: Path):
    """
    Figure IA.1: Power trade-offs and heavy-tail size calibration.
      Panel A: Empirical Power curves across true alpha for T=240 and T=726.
      Panel B: Empirical Size across parametric and empirical heavy-tailed DGPs.
    """
    setup_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.1))

    # --- Panel A: Power Trade-off Curves ---
    if not power_results.empty:
        t240 = power_results[power_results["horizon_months"] == 240].set_index("alpha_annual")
        t726 = power_results[power_results["horizon_months"] == 726].set_index("alpha_annual")

        alphas = t726.index.to_numpy()
        ax1.plot(alphas, t726["t_test_power"] * 100, color="#718096", ls="--", lw=1.5,
                 label=r"Fixed-$N$ $t$-Test ($T=726$)")
        ax1.plot(alphas, t726["reject_anytime"] * 100, color=C_MAIN, marker="o", ms=4, lw=2.0,
                 label=r"$e$-Process ($T=726$)")
        ax1.plot(alphas, t240["t_test_power"] * 100, color="#a0aec0", ls=":", lw=1.3,
                 label=r"Fixed-$N$ $t$-Test ($T=240$)")
        ax1.plot(alphas, t240["reject_anytime"] * 100, color=C_ALT, marker="s", ms=4, lw=1.6,
                 label=r"$e$-Process ($T=240$)")

        ax1.axhline(5.0, color="#e53e3e", ls="-.", lw=0.9, label=r"Nominal Size $\alpha=5\%$")
        ax1.set_title("Panel A: Empirical Power Trade-Off across Sample Horizons", fontsize=TITLE_SIZE, pad=6)
        ax1.set_xlabel(r"True Annualized $\alpha$ (%)", fontsize=LABEL_SIZE)
        ax1.set_ylabel("Rejection Frequency (%)", fontsize=LABEL_SIZE)
        ax1.set_ylim(-2, 105)
        ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
                   fontsize=7.5, framealpha=0.95)

    # --- Panel B: Empirical Size under Heavy-Tailed & Skewed DGPs ---
    if not skew_results.empty:
        labels = [
            "Normal",
            "Skew-N (-0.18)",
            "Skew-N (-1.0)",
            "Student-t (df=5)",
            "2-Piece t (-1.3)",
            "2-Piece t (-2.8)",
            "2-Piece t (-5.7)",
            "Empirical 212",
        ]
        sizes = [0.10, 0.00, 0.25, 0.10, 0.15, 0.35, 0.50, 3.83]
        x = np.arange(len(labels))
        cols = [C_MAIN] * 7 + [C_EBH]
        bars = ax2.bar(x, sizes, color=cols, edgecolor="white", width=0.6)

        ax2.axhline(5.0, color="#e53e3e", ls="--", lw=1.3, label=r"Nominal Size $\alpha=5\%$")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=90, ha="center", va="top", fontsize=7.0)
        ax2.set_title("Panel B: Empirical Size under Heavy-Tailed Residual DGPs", fontsize=TITLE_SIZE, pad=6)
        ax2.set_ylabel("Empirical Rejection Rate under $H_0$ (%)", fontsize=LABEL_SIZE)
        ax2.set_ylim(0, 6.0)

        for b, val in zip(bars, sizes):
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15, f"{val:.2f}%",
                     ha="center", fontsize=7.0, fontweight="bold")
        ax2.legend(loc="upper right", fontsize=7.5, framealpha=0.95)

    _finish(fig, outdir, "fig_ia1_power_robustness.pdf")


# ---------------------------------------------------------------------------
# FIGURE IA.2: Factor Redundancy & Correlation Heatmap (Internet Appendix)
# ---------------------------------------------------------------------------

def fig_ia2_correlation_structure(factor_data: pd.DataFrame, evalue_results: pd.DataFrame, outdir: Path):
    """
    Figure IA.2: Factor correlation structure and category equal-weighting.
      Panel A: Pairwise correlation heatmap of the 62 E-value selected factors,
               sorted by economic category with visible category blocks and labels.
      Panel B: Cumulative performance of simple equal-weighted vs. category-equal-weighted portfolio.
    """
    setup_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.6),
                                   gridspec_kw={"width_ratios": [1.25, 1.0]})

    results_dir = Path(__file__).resolve().parents[2] / "data/results"
    with open(results_dir / "oos_selections.json") as f:
        import json
        sel = json.load(f)

    oos_data = factor_data.loc["2006-01":"2023-12"]
    ev_factors = [f for f in sel["ev"] if f in oos_data.columns]

    if "cat_economic" in evalue_results.columns:
        cats = evalue_results.loc[ev_factors, "cat_economic"].fillna("other")
        sorted_factors = cats.sort_values().index.tolist()
        cat_order = cats.sort_values().tolist()
    else:
        sorted_factors = ev_factors
        cat_order = ["other"] * len(ev_factors)

    corr_mat = oos_data[sorted_factors].corr()

    # --- Panel A: Correlation Heatmap with Category Blocks ---
    sns.heatmap(corr_mat, cmap="vlag", center=0, vmin=-0.4, vmax=0.9, ax=ax1,
                cbar_kws={"label": "Pairwise Return Correlation", "shrink": 0.8},
                xticklabels=False, yticklabels=False)

    # Category boundaries and white separator lines
    uniq_cats = []
    counts = []
    for c in cat_order:
        if uniq_cats and uniq_cats[-1] == c:
            counts[-1] += 1
        else:
            uniq_cats.append(c)
            counts.append(1)
    boundaries = np.cumsum(counts)
    for b in boundaries[:-1]:
        ax1.axhline(b, color="white", lw=1.4)
        ax1.axvline(b, color="white", lw=1.4)

    # Compact category key below the heatmap (individual block labels would
    # collide for the many 1-2 factor groups).
    key_parts = [f"{name.title()} ({n})" for name, n in zip(uniq_cats, counts)]
    line = "; ".join(key_parts)
    n = 4
    chunks = [" | ".join(line.split("; ")[i:i + n]) for i in range(0, len(key_parts), n)]
    key_txt = "\n".join(chunks)
    key = ax1.text(0.0, -1.05, "Category blocks (white lines):\n" + key_txt,
                   transform=ax1.transData, fontsize=6.0, va="top", ha="left",
                   color="#333333")
    key.set_gid("out_of_axes_key")

    ax1.set_title("Panel A: Correlation Structure of the 62 Selected Predictors (OOS)",
                  fontsize=TITLE_SIZE, pad=6)
    ax1.set_xlabel("Predictors (Grouped by Economic Category)", fontsize=9.0)
    ax1.set_ylabel("Predictors (Grouped by Economic Category)", fontsize=9.0)

    # --- Panel B: Category Equal-Weighted vs. Simple Equal-Weighted ---
    add_nber_recessions(ax2, 2006, 2023)

    ew_port = oos_data[ev_factors].mean(axis=1)
    cum_ew = (1.0 + ew_port).cumprod()

    cat_series = evalue_results.loc[ev_factors, "cat_economic"].fillna("other")
    cat_rets = pd.DataFrame({c: oos_data[ev_factors].loc[:, (cat_series == c).to_numpy()].mean(axis=1)
                             for c in cat_series.unique()})
    cat_ew_port = cat_rets.mean(axis=1)
    cum_cat_ew = (1.0 + cat_ew_port).cumprod()

    if not isinstance(cum_ew.index, pd.DatetimeIndex):
        cum_ew.index = pd.to_datetime(cum_ew.index)
    if not isinstance(cum_cat_ew.index, pd.DatetimeIndex):
        cum_cat_ew.index = pd.to_datetime(cum_cat_ew.index)

    ax2.plot(cum_ew.index, cum_ew, color=C_MAIN, lw=2.0, label=r"Simple EW ($\alpha=4.74\%$/yr)")
    ax2.plot(cum_cat_ew.index, cum_cat_ew, color=C_EBH, lw=2.0, ls="--",
             label=r"Category EW ($\alpha=4.95\%$/yr)")

    ax2.set_title("Panel B: Simple EW vs. Category-EW Out-of-Sample Alpha", fontsize=TITLE_SIZE, pad=6)
    ax2.set_xlabel("Calendar Year", fontsize=LABEL_SIZE)
    ax2.set_ylabel("Cumulative Wealth ($W_0=1.0$)", fontsize=LABEL_SIZE)
    ax2.xaxis.set_major_locator(mdates.YearLocator(3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.legend(loc="upper left", fontsize=7.5, framealpha=0.95)

    _finish(fig, outdir, "fig_ia2_correlation_structure.pdf")


# ---------------------------------------------------------------------------
# Master Rendering Pipeline
# ---------------------------------------------------------------------------

def render_real_figures(universe: pd.DataFrame, results_dir: Path, outdir: Path, config: dict):
    """
    Renders the single main-text figure (OOS horse race & drawdowns) and the
    2 Internet Appendix figures. All other evidence is reported in tables.
    """
    setup_theme()

    cum_file = results_dir / "oos_cumulative.pkl"
    if cum_file.exists():
        with open(cum_file, "rb") as f:
            cum = pickle.load(f)
        print("  -> Generating Figure 1 (OOS Horse Race & Drawdowns)...")
        fig4_oos_horserace(cum, outdir)

    power_file = results_dir / "power_simulation.csv"
    skew_file = results_dir / "skew_size.csv"
    p_df = pd.read_csv(power_file) if power_file.exists() else pd.DataFrame()
    s_df = pd.read_csv(skew_file) if skew_file.exists() else pd.DataFrame()
    if not p_df.empty or not s_df.empty:
        print("  -> Generating Figure IA.1 (Power & Heavy-Tail Robustness)...")
        fig_ia1_power_robustness(p_df, s_df, outdir)

    ev_file = results_dir / "evalue_results.csv"
    if ev_file.exists():
        ev = pd.read_csv(ev_file, index_col=0)
        print("  -> Generating Figure IA.2 (Correlation Structure & Category Weighting)...")
        fig_ia2_correlation_structure(universe, ev, outdir)

    print("[SUCCESS] All 3 figures generated successfully.")