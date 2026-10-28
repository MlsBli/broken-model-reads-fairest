"""plot_paper_figures.py — Figures 1 and 2 of the paper, drawn from the saved results.

None of them needs a training run. Outputs go to analysis/results/figures/.

  bios_collapse_artifact.pdf      Deployed-head equalized odds against balanced
                                  accuracy on Bias in Bios: the lowest gaps
                                  belong to the models that stopped working.

  bios_budget_recovery.pdf        What a labelled test-time repair recovers on
                                  Bias in Bios, as a function of its budget.

Use the pinned Matplotlib (requirements.txt: matplotlib~=3.11.1). Output is
deterministic under that pin but not across minor versions: 3.10.6, for
example, renders a different dash phase and alpha compositing where the
threshold line crosses a fill_between band, which makes the alarm threshold
noticeably fainter.

Usage:
    python analysis/plot_paper_figures.py
"""

from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parent.parent
SPUR = ROOT / "spurious_shift"
BIOS = ROOT / "bias_in_bios"
OUT = ROOT / "analysis" / "results" / "figures"

sys.path.insert(0, str(SPUR))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import plot_style as ps  # noqa: E402

# Thresholds must match analysis/posthoc_studies.py (study E4).
ACC_TOL = 0.10
GAP_TOL = 0.15
COLLAPSE_FLOOR = 0.5  # the utility floor Table 2 uses to gate deployed-head EO

# The collapse figure is placed at 0.66\linewidth. plot_style assumes a figure
# is generated at the width it is placed at, so it is drawn at that width, with
# 9pt text to match the caption beside it.
PLACED_FRAC = 0.66
PLACED_W = PLACED_FRAC * 5.5
LABEL_PT = 9.0

# The four collapsed methods sit within 0.02 of each other on both axes, too
# close to label one by one, so their names are stacked beside the cluster in
# each method's colour behind a bracket. The rest are labelled directly at
# these (x, y, ha) positions.
COLLAPSED_STACK = ("fair_gem", "finetune", "agem", "group_dro")
DIRECT_LABELS = {
    "fairl":         (0.381, 0.124, "center"),
    "fair_derpp":    (0.659, 0.178, "center"),
    "fsw":           (0.622, 0.159, "right"),
    "derpp":         (0.622, 0.137, "right"),
    "joint_retrain": (0.745, 0.124, "center"),
}


def collapse_artifact() -> Path:
    with (BIOS / "results" / "results_test_summary.csv").open(encoding="utf-8") as fh:
        rows = {r["method"]: r for r in csv.DictReader(fh)}

    fig, ax = plt.subplots(figsize=(PLACED_W, 2.25))
    ax.axvspan(0, COLLAPSE_FLOOR, color="#D1495B", alpha=0.055, lw=0, zorder=0)
    ax.axvline(COLLAPSE_FLOOR, color="#D1495B", lw=0.8, ls=(0, (4, 2)), zorder=1)

    for method, row in rows.items():
        acc = float(row["BalAcc_mean"])
        eo = float(row["EO_mean"])
        ax.errorbar(acc, eo, xerr=float(row["BalAcc_std"]),
                    yerr=float(row["EO_std"]), fmt="none",
                    ecolor=ps.METHOD_COLORS.get(method, "0.4"),
                    elinewidth=0.7, alpha=0.65, zorder=2)
        ax.scatter(acc, eo, s=40 if method in ps.OURS else 30,
                   color=ps.METHOD_COLORS.get(method, "0.4"),
                   marker=ps.MARKERS.get(method, "o"),
                   edgecolor=ps.OURS_EDGE if method in ps.OURS else "0.25",
                   linewidth=1.0 if method in ps.OURS else 0.5, zorder=3)
        if method in DIRECT_LABELS:
            lx, ly, ha = DIRECT_LABELS[method]
            ax.annotate(ps.display_name(method), (lx, ly), fontsize=LABEL_PT,
                        color="0.2", ha=ha, va="center", zorder=4)

    stack_x, stack_top, stack_step = 0.262, 0.043, 0.021
    for i, method in enumerate(COLLAPSED_STACK):
        ax.annotate(ps.display_name(method),
                    (stack_x, stack_top - i * stack_step), fontsize=LABEL_PT,
                    color=ps.METHOD_COLORS[method], ha="left", va="center",
                    zorder=4)
    stack_bottom = stack_top - (len(COLLAPSED_STACK) - 1) * stack_step
    ax.plot([0.250, 0.244, 0.244, 0.250],
            [stack_top, stack_top, stack_bottom, stack_bottom],
            color="0.55", lw=0.6, solid_capstyle="butt", zorder=4)
    ax.plot([0.236, 0.244], [0.021, 0.021], color="0.55", lw=0.6, zorder=4)

    ax.set_xlabel("Deployed-head balanced accuracy $\\uparrow$", fontsize=9.5)
    ax.set_ylabel(ps.D_EO + " $\\downarrow$", fontsize=9.5)
    ax.tick_params(labelsize=LABEL_PT)
    ax.set_xlim(0.12, 0.88)
    ax.set_ylim(-0.032, 0.215)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticks([0.0, 0.05, 0.10, 0.15, 0.20])
    ax.annotate("collapsed:\nnear-zero gaps\nby artifact", (0.132, 0.209),
                fontsize=LABEL_PT, color="#D1495B", va="top", linespacing=1.1)
    ax.annotate("functioning models", (0.87, 0.209), fontsize=LABEL_PT,
                color="0.35", va="top", ha="right")
    ax.grid(True, alpha=0.18, lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    path = OUT / "bios_collapse_artifact.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def budget_recovery() -> Path:
    """The budget-recovery curve: what a labelled test-time repair buys.

    Two panels: (a) repair lifts a collapsed model at every budget, but leaves
    a working one below its own deployed head until the budget is large; (b)
    repair restores the disparity that collapse concealed.
    """
    rows = {}
    with (BIOS / "results" / "ttbudget_summary.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.setdefault(r["method"], []).append(r)
    for v in rows.values():
        v.sort(key=lambda r: float(r["budget"]))

    shown = ["agem", "derpp", "fair_derpp", "finetune"]
    fig, (ax_a, ax_g) = plt.subplots(1, 2, figsize=(ps.FULL_W, 2.35))

    for method in shown:
        rs = rows.get(method)
        if not rs:
            continue
        x = [max(float(r["budget"]), 12) for r in rs]   # 0 placed at the left edge
        style = dict(color=ps.METHOD_COLORS.get(method, "0.4"),
                     marker=ps.MARKERS.get(method, "o"), ms=3.0,
                     lw=1.6 if method in ps.OURS else 1.1,
                     label=ps.display_name(method))
        for ax, col in ((ax_a, "bal_acc"), (ax_g, "eo")):
            mu = [float(r[f"{col}_mean"]) for r in rs]
            sd = [float(r[f"{col}_std"]) for r in rs]
            ax.plot(x, mu, **style)
            ax.fill_between(x, [m - s for m, s in zip(mu, sd)],
                            [m + s for m, s in zip(mu, sd)], alpha=0.13,
                            color=ps.METHOD_COLORS.get(method, "0.4"), lw=0)
        # Deployed head as the line a repair has to beat, not just a point at B=0.
        ax_a.axhline(float(rs[0]["bal_acc_mean"]), lw=0.6, ls=(0, (1, 3)),
                     color=ps.METHOD_COLORS.get(method, "0.4"), alpha=0.7)

    for ax, ylab, title in (
            (ax_a, "balanced accuracy $\\uparrow$",
             "(a) What the repair recovers"),
            (ax_g, ps.D_EO + " $\\downarrow$",
             "(b) What the repair puts back")):
        ax.set_xscale("log")
        ax.set_xlabel("labelled examples per method ($B$; $0$ at left edge)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=7.4)
        ax.grid(True, alpha=0.18, lw=0.5)
        ax.set_axisbelow(True)
    ax_a.legend(fontsize=6.5, frameon=False, loc="lower right", ncol=1,
                handlelength=1.6, borderaxespad=0.4)
    fig.tight_layout(pad=0.35)
    path = OUT / "bios_budget_recovery.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


if __name__ == "__main__":
    ps.set_style()
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (collapse_artifact, budget_recovery):
        p = fn()
        print(f"wrote {p.relative_to(ROOT)}")
