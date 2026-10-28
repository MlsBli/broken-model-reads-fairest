"""
plot_ncm.py — Linear head vs. NCM balanced-accuracy comparison.

The Pareto frontier figure is produced by plot_results.py (plot_pareto). This
file holds only the nearest-class-mean diagnostic, which is bios-specific.

The gap between NCM and the deployed linear head reveals whether catastrophic
forgetting (e.g. in FAIR-GEM) is encoder forgetting or just head recency bias:
a large NCM gain means the representations were intact and only the head forgot.

Usage (run from bias_in_bios/results):
    python ../plot_ncm.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from plot_style import (apply_style, GRID_KW, value_colors, OURS_EDGE, FULL_W,
                        BAR_PAIR_WIDTH, BAR_EDGE, BAR_PAIR_EDGE_W, mark_our_bars)

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

apply_style()

# Method catalogue (same ordering as plot_results.py)
DISPLAY = {
    "joint_retrain":  "Joint Retrain",
    "finetune":       "Fine-tune",
    "agem":           "A-GEM",
    "derpp":          "DER++",
    "group_dro":      "GroupDRO",
    "fairl":          "FaIRL-insp.",
    "fair_gem":       "FAIR-GEM",
    "fair_derpp":     "Fair-DER++",
    "fsw":            "FSW",
}

METHOD_ORDER = [
    "joint_retrain",
    "finetune", "agem", "derpp",
    "group_dro", "fairl",
    "fsw",
    "fair_gem", "fair_derpp",
]


def _remap(d):
    return d


def plot_ncm_comparison():
    """Bar chart comparing linear head vs NCM balanced accuracy per method.

    Requires results_test.npz (from --reeval) and results_ncm.npy (from eval_ncm.py).
    The gap between NCM and linear head reveals whether catastrophic forgetting
    in methods like FAIR-GEM is due to encoder forgetting or just head recency bias.
    """
    ncm_agg_path = "results_ncm_full_agg.npz"
    ncm_path = "results_ncm.npy"
    test_path = ("results_test_agg.npz" if os.path.exists("results_test_agg.npz")
                 else "results_test.npz")
    if not os.path.exists(ncm_agg_path) and not os.path.exists(ncm_path):
        print("Skipping NCM plot - results_ncm.npy not found. Run eval_ncm.py first.")
        return
    if not os.path.exists(test_path):
        print("Skipping NCM plot - results_test.npz not found. Run train.py --reeval first.")
        return

    test_data = np.load(test_path, allow_pickle=True)
    bal_acc   = _remap(test_data["bal_acc"].item())
    N         = next(iter(bal_acc.values())).shape[0]
    if os.path.exists(ncm_agg_path):
        ncm_full = np.load(ncm_agg_path, allow_pickle=True)
        ncm_bal = _remap(ncm_full["bal_acc"].item())
        ncm_data = {
            method: matrix[N - 1, :N] for method, matrix in ncm_bal.items()
        }
    else:
        ncm_data = np.load(ncm_path, allow_pickle=True).item()

    methods = [m for m in METHOD_ORDER if m in ncm_data and m in bal_acc]
    x       = np.arange(len(methods))
    labels  = [DISPLAY.get(m, m) for m in methods]

    lin_vals = [float(np.nanmean([bal_acc[m][N - 1, e] for e in range(N)])) for m in methods]
    ncm_vals = [float(np.mean(ncm_data[m])) for m in methods]

    # Both bar groups are accuracy → green-by-value on a shared ramp.
    allv = [v for v in (lin_vals + ncm_vals) if np.isfinite(v)]
    vmin, vmax = (min(allv), max(allv)) if allv else (0.0, 1.0)
    fig, ax = plt.subplots(figsize=(FULL_W, 3.3), layout="constrained")
    w = BAR_PAIR_WIDTH
    bars_lin = ax.bar(x - w / 2, lin_vals, width=w,
                      color=value_colors(lin_vals, "acc", vmin=vmin, vmax=vmax), alpha=0.5,
                      edgecolor="0.5", linewidth=BAR_PAIR_EDGE_W)
    bars_ncm = ax.bar(x + w / 2, ncm_vals, width=w,
                      color=value_colors(ncm_vals, "acc", vmin=vmin, vmax=vmax), alpha=1.0,
                      edgecolor=BAR_EDGE, linewidth=BAR_PAIR_EDGE_W)

    mark_our_bars(bars_lin, methods)
    mark_our_bars(bars_ncm, methods)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Balanced accuracy\n(higher is better)")
    ax.set_title("Linear head vs. NCM evaluation\n"
                 "NCM gain indicates encoder-head separation")
    ax.grid(axis="y", **GRID_KW)
    ax.set_axisbelow(True)

    import matplotlib.patches as mp
    ax.legend(handles=[
        mp.Patch(facecolor="0.7", alpha=0.5,  edgecolor="0.5", label="Linear head"),
        mp.Patch(facecolor="0.4", alpha=1.0,  edgecolor="0.35", label="NCM (encoder only)"),
        mp.Patch(facecolor="0.85", hatch="////", edgecolor=OURS_EDGE, label="* Our methods"),
    ], loc="best")

    out = os.path.join(OUT_DIR, "ncm_comparison.pdf")
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    plot_ncm_comparison()
