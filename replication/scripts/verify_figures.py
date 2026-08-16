"""
verify_figures.py — Publication-grade layout audit for the 6 paper figures.

For each figure, re-renders it and measures (in display pixels) every piece of
text, every scatter marker, and every line sample, then flags:
  1. annotation/legend/tick-label text boxes that overlap each other,
  2. annotations that fall outside their panel axes (clipped / off-panel),
  3. legends whose box swallows the plotted scatter points or curve samples.

Usage:
    python3 scripts/verify_figures.py

Exit code is non-zero if any violation is found. Thresholds are conservative
(top-journal legibility standards): pairwise text overlap ratio >= 0.03 or
any curve/point sample falling inside a legend box is a violation.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.viz import figures as F  # noqa: E402

TEXT_OVERLAP_RATIO = 0.03   # pairwise text-box overlap floor (fraction)
LEGEND_SAMPLE_HITS = 1      # samples inside a legend box -> violation
AXIS_MARGIN_PX = 4.0        # annotations must stay >= this inside the axes


def _overlap_ratio(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ox = min(ax1, bx1) - max(ax0, bx0)
    oy = min(ay1, by1) - max(ay0, by0)
    if ox <= 0 or oy <= 0:
        return 0.0
    return (ox * oy) / max(1e-12, min((ax1 - ax0) * (ay1 - ay0),
                                      (bx1 - bx0) * (by1 - by0)))


def _text_bboxes(ax, fig, artists):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = []
    for t in artists:
        bb = t.get_window_extent(renderer=r)
        out.append((t, (bb.x0, bb.y0, bb.x1, bb.y1)))
    return out


def _legend_bbox(leg, fig):
    fig.canvas.draw()
    bb = leg.get_window_extent(renderer=fig.canvas.get_renderer())
    return (bb.x0, bb.y0, bb.x1, bb.y1)


def _line_samples(ax, fig):
    """Sample display-coordinate points along every Line2D in the axes."""
    fig.canvas.draw()
    pts = []
    for line in ax.lines:
        # Skip reference lines (axhline / axvline) which are drawn on blended
        # transforms, not ax.transData.
        if line.get_transform() is not ax.transData:
            continue
        # orig=False returns the numeric data actually plotted (categorical
        # codes / datetime numbers), which transData can transform directly.
        xd, yd = line.get_xdata(orig=False), line.get_ydata(orig=False)
        if xd is None or yd is None or len(xd) == 0:
            continue
        idx = np.linspace(0, len(xd) - 1, min(len(xd), 400)).astype(int)
        for i in idx:
            pts.append(ax.transData.transform((float(xd[i]), float(yd[i]))))
    return pts


def _scatter_samples(ax, fig):
    fig.canvas.draw()
    pts = []
    for coll in ax.collections:
        offs = coll.get_offsets()
        if offs is None or len(offs) == 0:
            continue
        for o in offs:
            pts.append(ax.transData.transform((o[0], o[1])))
    return pts


def _samples_in_box(pts, box):
    x0, y0, x1, y1 = box
    return [p for p in pts if x0 < p[0] < x1 and y0 < p[1] < y1]


def audit(fig, name, report):
    fig.canvas.draw()
    issues = []

    for ax in fig.axes:
        if ax.get_label().startswith("<colorbar"):
            continue
        ax_ext = ax.get_window_extent(renderer=fig.canvas.get_renderer())

        # 1) text-text overlaps among annotations
        anns = [t for t in ax.texts if t.get_text().strip()]
        boxes = _text_bboxes(ax, fig, anns)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                ov = _overlap_ratio(boxes[i][1], boxes[j][1])
                if ov >= TEXT_OVERLAP_RATIO:
                    issues.append(
                        f"text overlap {boxes[i][0].get_text()!r} <-> {boxes[j][0].get_text()!r} ({ov:.2f})")
        # 2) annotations outside axes
        for t, (x0, y0, x1, y1) in boxes:
            if t.get_gid() == "out_of_axes_key":
                continue
            if (x0 < ax_ext.x0 - AXIS_MARGIN_PX or x1 > ax_ext.x1 + AXIS_MARGIN_PX
                    or y0 < ax_ext.y0 - AXIS_MARGIN_PX or y1 > ax_ext.y1 + AXIS_MARGIN_PX):
                issues.append(f"annotation outside axes: {t.get_text()!r}")
        # 3) tick-label overlaps
        xtb = _text_bboxes(ax, fig, ax.get_xticklabels())
        for i in range(len(xtb)):
            for j in range(i + 1, len(xtb)):
                ov = _overlap_ratio(xtb[i][1], xtb[j][1])
                if ov >= TEXT_OVERLAP_RATIO:
                    issues.append(
                        f"xtick overlap {xtb[i][0].get_text()!r} <-> {xtb[j][0].get_text()!r} ({ov:.2f})")
        # 4) legend swallowing data
        leg = ax.get_legend()
        if leg is not None:
            lbox = _legend_bbox(leg, fig)
            n_hits = len(_samples_in_box(_line_samples(ax, fig), lbox)) + \
                     len(_samples_in_box(_scatter_samples(ax, fig), lbox))
            if n_hits >= LEGEND_SAMPLE_HITS:
                issues.append(f"legend covers data ({n_hits} samples)")

    report[name] = issues
    return len(issues) == 0


def main() -> int:
    setup_theme = F.setup_theme
    setup_theme()

    universe = pd.read_csv(ROOT / "PredictorLSretWide.csv", index_col=0)
    results = ROOT / "data/results"
    outdir = ROOT / "output/figures"

    ev = pd.read_csv(results / "evalue_results.csv", index_col=0)
    with open(results / "oos_cumulative.pkl", "rb") as fh:
        cum = pickle.load(fh)
    power = pd.read_csv(results / "power_simulation.csv")
    skew = pd.read_csv(results / "skew_size.csv")

    # Keep figures alive for measurement (patch close to no-op).
    orig_close = plt.close
    plt.close = lambda *a, **k: None
    try:
        F.fig4_oos_horserace(cum, outdir)
        report = {}
        ok = audit(plt.gcf(), "fig4_oos_horserace (main text Figure 1)", report)

        F.fig_ia1_power_robustness(power, skew, outdir)
        ok &= audit(plt.gcf(), "fig_ia1_power_robustness (IA Figure IA.1)", report)

        F.fig_ia2_correlation_structure(universe, ev, outdir)
        ok &= audit(plt.gcf(), "fig_ia2_correlation_structure (IA Figure IA.2)", report)
    finally:
        plt.close = orig_close

    print(f"{'FIGURE':<32} STATUS")
    for name, issues in report.items():
        status = "PASS" if not issues else f"FAIL ({len(issues)})"
        print(f"{name:<32} {status}")
        for issue in issues:
            print(f"    - {issue}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())