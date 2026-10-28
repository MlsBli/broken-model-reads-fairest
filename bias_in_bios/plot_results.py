"""
plot_results.py — Paper figures.

Reads results.npz (training) and results_test.npz / results_retune.npz (final eval).
Produces figures in figures/:
  summary_comparison.pdf  — bar chart: BalAcc, WorstGrp, AccParity, TPR
  forgetting_heatmaps.pdf — balanced accuracy lower-triangular matrices
  eo_dp_trajectory.pdf    — TPR gap per task (Spurious Shift only)
  head_vs_probe.pdf       — deployed head vs. retained-data head
  leakage.pdf             — gender leakage probe
  projection_diagnostics.pdf — safety projection firing rates

Fairness gaps are accuracy-weighted so a forgotten task (acc≈0) can't masquerade as fair.

Run from bias_in_bios/results:
    python ../plot_results.py [--test | --retune | --retune-fair]
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from plot_style import (apply_style, GRID_KW, value_colors, ACC_CMAP, FAIR_CMAP,
                        CAT_COLORS, OURS_EDGE, FULL_W, COL_W,
                        PARETO_HEIGHT_COL, heatmap_norm,
                        BAR_WIDTH, BAR_PAIR_WIDTH, BAR_EDGE, BAR_EDGE_W,
                        BAR_PAIR_EDGE_W, VALUE_LABEL_SIZE, mark_our_bars,
                        draw_pareto_panel, pareto_legend)

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

apply_style()


# Load
def _remap(d):
    return d


import sys

# Source for summary/Pareto bars: --test (deployed head, the primary protocol,
# and the default) | --retune (retained-data head) | --retune-fair
# (fairness-aware retained-data head).
_FINAL_SRC   = ("retune_fair" if "--retune-fair" in sys.argv
                else "retune" if "--retune" in sys.argv
                else "test")
_FINAL_PATHS = {
    "test": ("results_test_agg.npz" if os.path.exists("results_test_agg.npz")
             else "results_test.npz"),
    "retune": ("results_retune_agg.npz" if os.path.exists("results_retune_agg.npz")
                else "results_retune.npz"),
    "retune_fair": ("results_retune_fair_agg.npz"
                     if os.path.exists("results_retune_fair_agg.npz")
                     else "results_retune_fair.npz"),
}
_FINAL_LABEL = {
    "test":        "deployed linear head",
    "retune":      "memory-refit head (method-specific retained data)",
    "retune_fair": "fairness-aware memory-refit head (method-specific retained data)",
}[_FINAL_SRC]

if not os.path.exists(_FINAL_PATHS[_FINAL_SRC]):
    print(f"Skipping plot_results ({_FINAL_SRC}) - {_FINAL_PATHS[_FINAL_SRC]} not found.")
    sys.exit(0)

_TRAIN_PATH = "results_agg.npz" if os.path.exists("results_agg.npz") else "results.npz"
data      = np.load(_TRAIN_PATH, allow_pickle=True)
data_test = np.load(_FINAL_PATHS[_FINAL_SRC], allow_pickle=True)

states = data["states"].tolist()
N      = len(states)

bal_acc      = _remap(data["bal_acc"].item())
acc          = _remap(data["acc"].item())
eo           = _remap(data["eo"].item())              # true equalized-odds gap
dp           = _remap(data["dp"].item()) if "dp" in data else None
tpr          = _remap(data["tpr"].item()) if "tpr" in data else None
acc_parity   = _remap(data["acc_parity"].item()) if "acc_parity" in data else None
worst_group  = _remap(data["worst_group"].item()) if "worst_group" in data else None

bal_acc_test = _remap(data_test["bal_acc"].item())
acc_test     = _remap(data_test["acc"].item())
eo_test      = _remap(data_test["eo"].item())
dp_test      = _remap(data_test["dp"].item()) if "dp" in data_test else None
tpr_test     = _remap(data_test["tpr"].item()) if "tpr" in data_test else None
acc_parity_test = _remap(data_test["acc_parity"].item()) if "acc_parity" in data_test else None
worst_group_test = _remap(data_test["worst_group"].item()) if "worst_group" in data_test else None


# Task labels and shortcut/legit schedules
def _parse_state(s):
    """Parse a task-state string. Handles two dataset formats:
      BIOS:     'task_0__professions=0-4'              → professions='0-4'
      Spurious: 'task_i__legit=X__shortcut=Y'          → legit/shortcut set
    Returns (task_idx_str, legit_or_None, shortcut, professions_or_None)."""
    task_idx = s.split("__")[0].replace("task_", "")
    sc, ls, prof = 0.0, None, None
    for part in s.split("__"):
        if part.startswith("shortcut="):
            sc = float(part.split("=")[1])
        elif part.startswith("legit="):
            ls = float(part.split("=")[1])
        elif part.startswith("professions="):
            prof = part.split("=")[1]
    return task_idx, ls, sc, prof


# Dataset detection: BIOS task states carry a 'professions=' tag; the synthetic
# spurious_shift states carry 'shortcut='. This drives whether the figures use
# profession labels (BIOS) or shortcut/legit annotations (spurious_shift).
IS_BIOS = any(_parse_state(s)[3] is not None for s in states)


def _task_label(s, short=False):
    """BIOS:     'task_0__professions=0-4'      → 'Task 0\n(prof 0-4)'  / 'T0'
       Spurious: 'task_2__legit=1.5__shortcut=3' → 'Task 2\n(l=1.5, s=3)' / 'T2'."""
    idx, ls, sc, prof = _parse_state(s)
    prefix = "T" if short else "Task "
    if prof is not None:
        return f"{prefix}{idx}" if short else f"{prefix}{idx}\n(prof {prof})"
    if ls is not None:
        return f"{prefix}{idx}\n(l={ls:g}, s={sc:g})"
    return f"{prefix}{idx}\n(s={sc:g})"


XLABS       = [_task_label(s)             for s in states]
XLABS_SHORT = [_task_label(s, short=True) for s in states]

SHORTCUTS = [_parse_state(s)[2] for s in states]
LEGITS    = [_parse_state(s)[1] for s in states]   # None if no legit tag

# Region where shortcut dominates legitimate signal (shortcut > legit).
# BIOS has no shortcut schedule, so this stays empty and no arc is drawn.
SHORTCUT_DOM_IDXS = [
    i for i, (sc, ls) in enumerate(zip(SHORTCUTS, LEGITS))
    if ls is not None and sc > ls
]


def _annotate_shortcut_arc(ax):
    """Shade the region where the shortcut dominates the legitimate signal.
    No-op for BIOS (class-incremental, no spurious-correlation schedule)."""
    if IS_BIOS:
        return
    if not SHORTCUT_DOM_IDXS:
        # No legit schedule: shade the peak shortcut region instead
        max_sc = max(SHORTCUTS)
        peak = [i for i, v in enumerate(SHORTCUTS) if v >= max_sc]
        lo_i, hi_i = min(peak) - 0.45, max(peak) + 0.45
        label = f"peak shortcut\n(s={max_sc:g})"
    else:
        lo_i = min(SHORTCUT_DOM_IDXS) - 0.45
        hi_i = max(SHORTCUT_DOM_IDXS) + 0.45
        label = "shortcut\ndominates"
    ax.axvspan(lo_i, hi_i, alpha=0.07, color="#d4781a", zorder=0)
    ylo, yhi = ax.get_ylim()
    ax.text((lo_i + hi_i) / 2, yhi - (yhi - ylo) * 0.04,
            label, ha="center", va="top", fontsize=6.5, color="#9b5000")


# Method catalogue
DISPLAY = {
    "joint_retrain":  "Joint Retrain†",
    "finetune":       "Fine-tune",
    "agem":           "A-GEM",
    "derpp":          "DER++",
    "group_dro":      "GroupDRO",
    "fairl":          "FaIRL-insp.",
    "fsw":            "FSW",
    "fair_gem":       "FAIR-GEM *",
    "fair_derpp":     "Fair-DER++ *",
}

COLORS = CAT_COLORS   # shared harmonised categorical palette (plot_style.py)

OURS   = {"fair_gem", "fair_derpp"}
ORACLE = "joint_retrain"

# Ordered for summary bar chart (all methods)
ALL_METHODS_ORD = [
    "joint_retrain",
    "finetune", "agem", "derpp",
    "group_dro", "fairl", "fsw",
    "fair_gem", "fair_derpp",
]

# Ordered for heatmaps and trajectories (CL methods only)
CL_METHODS_ORD = [
    "joint_retrain", "finetune", "agem", "derpp",
    "group_dro", "fairl", "fsw", "fair_gem", "fair_derpp",
]

SUMMARY_METHODS = [m for m in ALL_METHODS_ORD if m in bal_acc_test]
CL_METHODS      = [m for m in CL_METHODS_ORD  if m in bal_acc]

TRAJ_STYLES = {
    "joint_retrain": dict(label="Joint Retrain†", color=COLORS["joint_retrain"], lw=1.5, ls="--", marker="s", ms=3.5, zorder=2),
    "finetune":      dict(label="Fine-tune",      color=COLORS["finetune"],      lw=1.3, ls="-",  marker="o", ms=3,   zorder=2),
    "agem":          dict(label="A-GEM",          color=COLORS["agem"],          lw=1.3, ls="-",  marker="o", ms=3,   zorder=2),
    "derpp":         dict(label="DER++",          color=COLORS["derpp"],         lw=1.3, ls="-",  marker="D", ms=3,   zorder=2),
    "group_dro":     dict(label="GroupDRO",       color=COLORS["group_dro"],     lw=1.3, ls="-.", marker="o", ms=3,   zorder=2),
    "fairl":         dict(label="FaIRL-insp.",    color=COLORS["fairl"],         lw=1.3, ls="-",  marker="v", ms=3,   zorder=2),
    "fsw":           dict(label="FSW",            color=COLORS["fsw"],           lw=1.3, ls=":",  marker="P", ms=3,   zorder=2),
    "fair_gem":      dict(label="FAIR-GEM *",     color=COLORS["fair_gem"],      lw=2.3, ls="-",  marker="o", ms=4,   zorder=3),
    "fair_derpp":    dict(label="Fair-DER++ *",   color=COLORS["fair_derpp"],    lw=2.3, ls="-",  marker="^", ms=4,   zorder=3),
}


# Scalar helpers
def _final_avg_test(mat):
    """Mean across all tasks in the final row of the test matrix."""
    return float(np.nanmean(mat[N - 1, :]))


# Below MIN_VALID_ACC, a task's predictions have collapsed (the model predicts
# one recent class for everyone) and any group-fairness gap on it is meaningless:
# both groups score ~0, so the gap reads as ~0 ("fair by being useless"). Fairness
# summaries must exclude / down-weight these tasks, else catastrophic forgetting
# masquerades as fairness — see finetune, which shows EO~0 only where acc~0.
MIN_VALID_ACC = 0.10


def _acc_weighted_final(bal_mat, fair_mat):
    """Final-row fairness gap, weighted by each task's final-model accuracy.
    Tasks the model has forgotten (acc≈0) contribute ≈0 weight, so a collapsed
    prediction cannot be scored as 'fair'. (Same construction as plot_pareto.)"""
    accs = np.array([bal_mat[N - 1, e] for e in range(N)])
    fair = np.array([fair_mat[N - 1, e] for e in range(N)])
    mask = ~(np.isnan(accs) | np.isnan(fair))
    if mask.sum() == 0:
        return float("nan")
    w = np.clip(accs[mask], 0, None)
    if w.sum() < 1e-8:
        return float(np.nanmean(fair[mask]))  # nothing retained, fall back to unweighted mean
    return float(np.average(fair[mask], weights=w))


def _bwt(bal_acc_mat):
    """Accuracy BWT = mean of bal_acc[T-1,i] - bal_acc[i,i] for i < T-1.
    Negative = accuracy forgotten on old tasks. Positive = backward transfer."""
    vals = [bal_acc_mat[N - 1, i] - bal_acc_mat[i, i]
            for i in range(N - 1)
            if not np.isnan(bal_acc_mat[N - 1, i]) and not np.isnan(bal_acc_mat[i, i])]
    return float(np.mean(vals)) if vals else float("nan")


def _f_bwt(fair_mat, bal_mat=None):
    """Fairness BWT: mean of fair[i,i] - fair[T-1,i] for i < T-1.
    Positive = fairness on past tasks improved (gap shrank). Negative = forgot.

    If bal_mat is given, tasks whose FINAL-model accuracy has collapsed
    (bal[T-1,i] < MIN_VALID_ACC) are excluded: their final-row gap is ~0 only
    because the model forgot them, which would otherwise register as a spurious
    fairness *improvement*. This is exactly the finetune artifact — without the
    guard, catastrophic forgetting reads as positive F-BWT."""
    vals = []
    for i in range(N - 1):
        if np.isnan(fair_mat[N - 1, i]) or np.isnan(fair_mat[i, i]):
            continue
        if bal_mat is not None and (np.isnan(bal_mat[N - 1, i])
                                    or bal_mat[N - 1, i] < MIN_VALID_ACC):
            continue
        vals.append(fair_mat[i, i] - fair_mat[N - 1, i])
    return float(np.mean(vals)) if vals else float("nan")


def _tight_ylim(vals, margin=0.25, anchor_zero=False):
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return 0.0, 1.0
    lo, hi = min(vals), max(vals)
    if anchor_zero:
        # Bars must rise from the axis baseline, so pin the lower limit at zero
        # exactly (never below it) — no bar floats above the x-axis.
        lo = min(lo, 0.0)
        span = (hi - lo) if (hi - lo) > 1e-6 else abs(hi) * 0.2 + 0.01
        return lo, hi + margin * span
    span = (hi - lo) if (hi - lo) > 1e-6 else abs(hi) * 0.2 + 0.01
    return lo - margin * span, hi + margin * span


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Summary comparison bar chart (held-out test results)
# ══════════════════════════════════════════════════════════════════════════════

def plot_summary():
    methods = SUMMARY_METHODS
    x       = np.arange(len(methods))
    labels  = [DISPLAY.get(m, m) for m in methods]
    colors  = [COLORS[m] for m in methods]

    bal_vals  = [_final_avg_test(bal_acc_test[m]) for m in methods]
    bwt_vals  = [_bwt(acc[m])                     for m in methods]
    # Worst-group balanced accuracy — the Rawlsian/DRO objective our methods
    # target. Higher is better; it is an accuracy (not a gap), so it is reported
    # as the unweighted final-row mean, exactly like balanced accuracy.
    worst_vals = ([_final_avg_test(worst_group_test[m]) for m in methods]
                  if worst_group_test is not None else None)
    # Fairness gaps are ACCURACY-WEIGHTED: a task the model forgot (acc≈0)
    # contributes ≈0 weight, so a collapsed prediction cannot be scored as fair.
    eo_vals   = [_acc_weighted_final(bal_acc_test[m], eo_test[m]) for m in methods]
    tpr_vals  = ([_acc_weighted_final(bal_acc_test[m], tpr_test[m]) for m in methods]
                 if tpr_test is not None else None)
    dp_vals   = ([_acc_weighted_final(bal_acc_test[m], dp_test[m]) for m in methods]
                 if dp_test is not None else None)
    accpar_vals = ([_acc_weighted_final(bal_acc_test[m], acc_parity_test[m]) for m in methods]
                   if acc_parity_test is not None else None)
    # Fairness BWT on the PRIMARY metric (TPR gap if available, else EO),
    # with forgotten tasks excluded so forgetting can't read as a fairness gain.
    if tpr is not None:
        fbwt_vals = [_f_bwt(tpr[m], bal_acc[m]) for m in methods]
        fbwt_lbl  = "Fairness BWT (TPR gap)\n(higher is better, 0 = no change)"
    else:
        fbwt_vals = [_f_bwt(eo[m], bal_acc[m]) for m in methods]
        fbwt_lbl  = "Fairness BWT (EO gap)\n(higher is better, 0 = no change)"

    # Specs: (values, ylabel, anchor_zero, kind).
    # kind="acc" → green-by-value; kind="gap" → blue-by-value (lower = fairer).
    specs = [
        (bal_vals, r"Balanced accuracy $\uparrow$",                True, "acc"),
    ]
    if worst_vals is not None:
        specs.append((worst_vals,
            r"Worst-group balanced accuracy $\uparrow$", True, "acc"))
    if accpar_vals is not None:
        specs.append((accpar_vals,
            r"Accuracy parity gap $\downarrow$", True, "gap"))

    # Single row of focused panels (method identity, incl. */† tags, is read off
    # the x-axis labels). Each bar is annotated with its value for quick reading.
    n_panels = len(specs)
    fig, axes = plt.subplots(1, n_panels, figsize=(FULL_W, 3.5), squeeze=False,
                             layout="constrained")
    axes_flat = axes.flatten()

    oracle_idx = methods.index(ORACLE) if ORACLE in methods else None

    for ax, (vals, ylabel, anchor_zero, kind) in zip(axes_flat, specs):
        ax.set_axisbelow(True)
        ax.grid(axis="y", **GRID_KW)

        bars = ax.bar(x, vals, color=value_colors(vals, kind), width=BAR_WIDTH,
                      zorder=2, edgecolor=BAR_EDGE, linewidth=BAR_EDGE_W)

        mark_our_bars(bars, methods)

        # Headroom for the value labels.
        lo, hi = _tight_ylim(vals, anchor_zero=anchor_zero)
        ax.set_ylim(lo, hi + 0.10 * (hi - lo))

        if oracle_idx is not None:
            ax.axhline(vals[oracle_idx], color="0.45", lw=0.9,
                       ls="--", zorder=1, alpha=0.8)

        # Value label atop each bar (2 dp for accuracies, 3 dp for gaps).
        fmt = "{:.3f}" if kind == "gap" else "{:.2f}"
        ylo, yhi = ax.get_ylim()
        for i, v in enumerate(vals):
            if np.isnan(v):
                continue
            ax.text(i, v + 0.015 * (yhi - ylo), fmt.format(v).lstrip("0"),
                    ha="center", va="bottom", fontsize=VALUE_LABEL_SIZE,
                    color=(OURS_EDGE if methods[i] in OURS else "0.25"),
                    fontweight="bold" if methods[i] in OURS else "normal",
                    zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.6, len(methods) - 0.4)
        ax.margins(x=0.01)

    # Legend explains the colour SEMANTICS (green = accuracy, blue = fairness),
    # not per-method identity (methods are read off the x-axis).
    legend_elements = [
        mpatches.Patch(facecolor=ACC_CMAP(0.85), edgecolor="0.35",
                       label="green = accurate"),
        mpatches.Patch(facecolor=FAIR_CMAP(0.85), edgecolor="0.35",
                       label="blue = fairer"),
        mpatches.Patch(facecolor="0.85", hatch="////", edgecolor=OURS_EDGE,
                       label="* our methods"),
        Line2D([0], [0], color="0.45", lw=0.9, ls="--",
               label="† joint-training level"),
    ]
    fig.legend(handles=legend_elements, loc="outside lower center", ncol=4,
               frameon=True, edgecolor="0.80")
    if IS_BIOS:
        src = "memory-refit head" if _FINAL_SRC != "test" else "deployed head"
        title = (f"Accuracy and fairness of the final model across all {N} "
                 f"Bias-in-Bios tasks")
        subtitle = f"held-out test set, {src}"
    else:
        sc_str = ", ".join(f"{v:g}" for v in SHORTCUTS)
        title = (f"Accuracy and fairness of the final model across all {N} "
                 f"Spurious-Shift tasks")
        subtitle = f"shortcut schedule [{sc_str}]"
    # Title via suptitle so constrained_layout reserves space (no manual coords
    # that could overlap/clip). The subtitle lives in the LaTeX caption.
    fig.suptitle(title, fontweight="semibold")
    # Suffixed by source (--test / --retune / --retune-fair) so the three
    # variants do not overwrite one another.
    out = os.path.join(OUT_DIR, f"summary_comparison_{_FINAL_SRC}.pdf")
    fig.savefig(out)
    print(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Forgetting heatmaps (balanced accuracy lower-triangular matrices)
# ══════════════════════════════════════════════════════════════════════════════

def plot_heatmaps():
    # Balanced-accuracy lower-triangular matrices, robust 5–95 pct Greens norm, equal aspect,
    # spineless cells, shared colourbar.
    hmap_methods = [m for m in CL_METHODS if m in bal_acc]
    ncols   = min(4, len(hmap_methods))
    nrows   = (len(hmap_methods) + ncols - 1) // ncols

    # Full text WIDTH (placed at \linewidth): up to 4 methods per row; cells square.
    _panel = FULL_W / ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(FULL_W, _panel * nrows + 0.5 * nrows),
                             constrained_layout=True)
    axes_flat = np.atleast_1d(axes).flatten()

    all_vals = np.concatenate([
        [bal_acc[m][t, e] for t in range(N) for e in range(t + 1)]
        for m in hmap_methods])
    norm = heatmap_norm(all_vals)
    cmap = ACC_CMAP
    im_ref = None

    for ax, m in zip(axes_flat, hmap_methods):
        mat = np.full((N, N), np.nan)
        for t in range(N):
            for e in range(t + 1):
                mat[t, e] = bal_acc[m][t, e]

        im_ref = ax.imshow(mat, norm=norm, cmap=cmap, aspect="equal")

        for t in range(N):
            for e in range(t + 1):
                v = mat[t, e]
                if np.isfinite(v):
                    tcol = "white" if norm(v) > 0.6 else "0.15"
                    ax.text(e, t, f"{v:.2f}", ha="center", va="center",
                            fontsize=8.0, color=tcol)

        ax.set_xticks(range(N)); ax.set_xticklabels(XLABS_SHORT)
        ax.set_yticks(range(N)); ax.set_yticklabels(XLABS_SHORT)
        ax.set_ylabel("after training on", fontsize=8.0)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)

        avg = float(np.nanmean(bal_acc[m][N - 1, :N]))
        ax.set_title(f"{DISPLAY.get(m, m)} (avg {avg:.2f})", fontsize=8.0)

    for ax in axes_flat[len(hmap_methods):]:
        ax.set_visible(False)

    if im_ref is not None:
        cbar = fig.colorbar(im_ref, ax=axes_flat[:len(hmap_methods)].tolist(),
                            fraction=0.020, pad=0.02)
        cbar.set_label("balanced accuracy  (greener = more accurate)", fontsize=8.0)
        cbar.outline.set_visible(False)

    fig.suptitle("Per-task balanced accuracy after sequential training",
                 fontsize=9.0)
    out = os.path.join(OUT_DIR, "forgetting_heatmaps.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — TPR gap (primary) on current task (diagonal) with shortcut arc
# ══════════════════════════════════════════════════════════════════════════════

def plot_trajectories():
    """Three-panel trajectory figure (Spurious Shift only — skipped on BIOS).

    (a) Task-0 accuracy over time — forgetting proxy.
    (b) Average accuracy over all seen tasks — adaptability.
    (c) TPR gap on the current task (diagonal) — current-task fairness.
    """
    if IS_BIOS:
        print("Skipping eo_dp_trajectory.pdf - Spurious-Shift figure; "
              "current data is BIOS (use head_vs_probe / summary instead).")
        return

    x     = np.arange(N)
    shown = [m for m in TRAJ_STYLES if m in eo]

    # Columns: (data_dict, ylabel, title, anchor_zero, extractor)
    #   extractor(mat) → 1-D array of length N to plot
    col_specs = [
        (acc,
         r"Task-0 accuracy $\uparrow$",
         "(a) Task-0 accuracy (forgetting)",
         False,
         lambda mat: np.array([mat[t, 0] for t in range(N)])),
        (acc,
         r"Avg accuracy, seen tasks $\uparrow$",
         "(b) Avg accuracy (adaptability)",
         False,
         lambda mat: np.array([np.nanmean(mat[t, :t + 1]) for t in range(N)])),
        # Panel (c): fairness gap on the CURRENT (just-trained) task — the
        # diagonal mat[t,t]. Evaluated right after training, so accuracy is high
        # and the gap is meaningful (not the collapse-induced ~0 of forgotten
        # tasks). Uses TPR gap — the primary BIOS metric — when available.
        (tpr if tpr is not None else eo,
         (r"TPR gap, current task $\downarrow$" if tpr is not None
          else r"EO gap, current task $\downarrow$"),
         ("(c) TPR gap (current task)" if tpr is not None
          else "(c) EO gap (current task)"),
         True,
         lambda mat: np.array([mat[t, t] for t in range(N)])),
    ]

    n_cols = len(col_specs)
    fig, axes = plt.subplots(1, n_cols, figsize=(FULL_W, 3.2),
                             constrained_layout=True)
    if n_cols == 1:
        axes = [axes]

    for ax, (metric, ylabel, title, anchor_zero, extractor) in zip(axes, col_specs):
        ax.set_axisbelow(True)
        ax.grid(axis="y", **GRID_KW)

        diag = {}
        for m in shown:
            if m not in metric:
                continue
            diag[m] = extractor(metric[m])
            s = TRAJ_STYLES[m]
            ax.plot(x, diag[m],
                    label=s["label"], color=s["color"],
                    lw=s["lw"], ls=s["ls"],
                    marker=s["marker"], markersize=s["ms"],
                    zorder=s["zorder"])

        all_vals = [v for d in diag.values() for v in d if not np.isnan(v)]
        if all_vals:
            ax.set_ylim(*_tight_ylim(all_vals, anchor_zero=anchor_zero))

        _annotate_shortcut_arc(ax)

        ax.set_xticks(x)
        ax.set_xticklabels(XLABS_SHORT)
        ax.set_xlabel("Training task")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(-0.4, N - 0.6)
        ax.legend(loc="best", ncol=2, fontsize=8.5)

    footer = ("† Joint training uses all observed data (plain ERM).  * Our methods.")
    if not IS_BIOS:
        footer += "  Shaded: shortcut dominates legitimate signal."
    # supxlabel is reserved by constrained_layout (stays inside the canvas).
    fig.supxlabel(footer, fontsize=8.5, color="0.4")

    out = os.path.join(OUT_DIR, "eo_dp_trajectory.pdf")
    fig.savefig(out)
    print(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Deployed linear head vs. retained-data head (head recency bias)
# ══════════════════════════════════════════════════════════════════════════════

def plot_head_vs_probe():
    """Compare the deployed head with a head refit on method-specific retained data.

    This diagnoses class-incremental head recency, but replay methods use their
    own buffers while memoryless methods use matched-size samples. It therefore
    measures recoverability from retained data, not encoder quality alone.

    Requires results_retune.npz (from retune_head.py) and results_test.npz
    (from train.py --reeval). Skips gracefully if either is missing.
    """
    dep_path = ("results_test_agg.npz" if os.path.exists("results_test_agg.npz")
                else "results_test.npz")
    probe_path = ("results_retune_agg.npz" if os.path.exists("results_retune_agg.npz")
                  else "results_retune.npz")
    if not os.path.exists(probe_path):
        print("Skipping head-vs-probe plot - results_retune.npz not found. "
              "Run retune_head.py first.")
        return
    if not os.path.exists(dep_path):
        print("Skipping head-vs-probe plot - results_test.npz not found. "
              "Run train.py --reeval first.")
        return

    dep = np.load(dep_path, allow_pickle=True)
    prb = np.load(probe_path, allow_pickle=True)

    dep_bal = _remap(dep["bal_acc"].item()); prb_bal = _remap(prb["bal_acc"].item())
    dep_eo  = _remap(dep["eo"].item());      prb_eo  = _remap(prb["eo"].item())
    dep_dp  = (_remap(dep["dp"].item())  if "dp"  in dep else None)
    prb_dp  = (_remap(prb["dp"].item())  if "dp"  in prb else None)
    dep_tpr = (_remap(dep["tpr"].item()) if "tpr" in dep else None)
    prb_tpr = (_remap(prb["tpr"].item()) if "tpr" in prb else None)
    dep_ap  = (_remap(dep["acc_parity"].item()) if "acc_parity" in dep else None)
    prb_ap  = (_remap(prb["acc_parity"].item()) if "acc_parity" in prb else None)
    dep_wg  = (_remap(dep["worst_group"].item()) if "worst_group" in dep else None)
    prb_wg  = (_remap(prb["worst_group"].item()) if "worst_group" in prb else None)

    methods = [m for m in CL_METHODS_ORD if m in dep_bal and m in prb_bal]
    x       = np.arange(len(methods))
    labels  = [DISPLAY.get(m, m) for m in methods]
    cols    = [COLORS[m] for m in methods]

    def favg(mat):   # final-row mean (accuracy panel)
        return float(np.nanmean(mat[N - 1, :]))

    def fwavg(bmat, fmat):  # accuracy-weighted fairness gap, per side
        return _acc_weighted_final(bmat, fmat)

    # All available panels, keyed; kind drives green/blue semantics. Gaps are
    # accuracy-weighted on BOTH sides so the deployed head's collapse-induced ~0
    # gaps can't read as spuriously fairer than the probe.
    P = {"bal": (r"Balanced accuracy $\uparrow$", "acc",
                 [favg(dep_bal[m]) for m in methods],
                 [favg(prb_bal[m]) for m in methods])}
    if dep_ap is not None and prb_ap is not None:
        P["accpar"] = (r"Accuracy parity gap $\downarrow$", "gap",
                       [fwavg(dep_bal[m], dep_ap[m]) for m in methods],
                       [fwavg(prb_bal[m], prb_ap[m]) for m in methods])
    if dep_wg is not None and prb_wg is not None:
        P["worst"] = (r"Worst-group balanced accuracy $\uparrow$", "acc",
                      [favg(dep_wg[m]) for m in methods],
                      [favg(prb_wg[m]) for m in methods])
    if dep_tpr is not None and prb_tpr is not None:
        P["tpr"] = (r"TPR gap $\downarrow$", "gap",
                    [fwavg(dep_bal[m], dep_tpr[m]) for m in methods],
                    [fwavg(prb_bal[m], prb_tpr[m]) for m in methods])
    if dep_dp is not None and prb_dp is not None:
        P["dp"] = (r"Demographic parity gap $\downarrow$", "gap",
                   [fwavg(dep_bal[m], dep_dp[m]) for m in methods],
                   [fwavg(prb_bal[m], prb_dp[m]) for m in methods])

    w = BAR_PAIR_WIDTH

    def _render(keys, outname, suptitle, width):
        keys = [k for k in keys if k in P]
        if not keys:
            return
        n = len(keys)
        # Generated at its placement width so figure text renders ~1:1 in the
        # PDF (COL_W for the single-column body figure, FULL_W for the appendix).
        narrow = width <= COL_W + 0.1   # single-column: vertical x-labels so the
                                        # 8 method names don't overlap.
        fig, axes = plt.subplots(1, n, figsize=(width, 2.4 if narrow else 3.4),
                                 squeeze=False, layout="constrained")
        axes_flat = axes.flatten()
        for ax, k in zip(axes_flat, keys):
            ylabel, kind, dep_vals, prb_vals = P[k]
            ax.set_axisbelow(True)
            ax.grid(axis="y", **GRID_KW)
            allv = [v for v in (dep_vals + prb_vals) if np.isfinite(v)]
            vmin, vmax = (min(allv), max(allv)) if allv else (0.0, 1.0)
            b1 = ax.bar(x - w / 2, dep_vals, width=w,
                        color=value_colors(dep_vals, kind, vmin=vmin, vmax=vmax),
                        alpha=0.45, edgecolor="0.5", linewidth=BAR_PAIR_EDGE_W)
            b2 = ax.bar(x + w / 2, prb_vals, width=w,
                        color=value_colors(prb_vals, kind, vmin=vmin, vmax=vmax),
                        alpha=1.0, edgecolor=BAR_EDGE, linewidth=BAR_PAIR_EDGE_W)
            mark_our_bars(b1, methods)
            mark_our_bars(b2, methods)
            ax.set_xticks(x)
            if narrow:
                ax.set_xticklabels(labels, rotation=90, ha="center")
            else:
                ax.set_xticklabels(labels, rotation=40, ha="right")
            ax.set_ylabel(ylabel)

        legend_elements = [
            mpatches.Patch(facecolor="0.75", alpha=0.55, edgecolor="0.5",
                           label="deployed head (lighter)"),
            mpatches.Patch(facecolor="0.45", alpha=1.0, edgecolor="0.35",
                           label="memory-refit head (solid)"),
            mpatches.Patch(facecolor="0.85", hatch="////", edgecolor=OURS_EDGE,
                           label="* our methods"),
        ]
        # Legend inside the canvas (constrained_layout reserves space) and
        # wrapped to fit the column width, so the saved PDF stays exactly figsize.
        fig.legend(handles=legend_elements, loc="outside lower center",
                   ncol=2 if narrow else 3, frameon=True, edgecolor="0.80")
        fig.suptitle(suptitle)
        out = os.path.join(OUT_DIR, outname)
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")

    # Main figure (single column): balanced accuracy + accuracy parity gap.
    _render(["bal", "accpar"], "head_vs_probe.pdf",
            "Deployed head vs. memory-refit head", COL_W)
    # Appendix figure (full width): worst-group, TPR, DP.
    _render(["worst", "tpr", "dp"], "head_vs_probe_appendix.pdf",
            "Deployed head and memory-refit head: worst-group accuracy, TPR & DP gaps",
            FULL_W)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Gender leakage probe (representation fairness)
# ══════════════════════════════════════════════════════════════════════════════

def plot_leakage():
    """Per-method gender-leakage: accuracy of a logistic-regression probe trained
    to recover the sensitive attribute (gender) from each method's penultimate
    representation. Higher = more gender information encoded = less fair
    representation; chance (0.5) is the fair ideal. The frozen-BERT input probe
    is the shared baseline (leakage before any CL training).

    This is the representation-level fairness view that the prediction-level EO/TPR
    gaps cannot show — and unlike accuracy-parity it is NOT gamed by collapse.

    Requires results_leakage.npy (from probe_leakage.py). Skips if missing.
    """
    audit_path = os.path.join("..", "..", "analysis", "results", "probe_diagnostics.csv")
    audit_rows = []
    if os.path.exists(audit_path):
        with open(audit_path, newline="", encoding="utf-8") as fh:
            audit_rows = [
                row for row in csv.DictReader(fh)
                if row.get("benchmark") == "bios" and row.get("study") == "leakage"
            ]
    if not audit_rows and not os.path.exists("results_leakage.npy"):
        print("Skipping leakage plot - results_leakage.npy not found. "
              "Run probe_leakage.py first.")
        return

    if audit_rows:
        values = {}
        for row in audit_rows:
            values.setdefault((row["method"], row["metric"]), []).append(float(row["mean"]))
        base_acc = float(np.mean(values[("frozen_bert_input", "acc")]))
        base_auc = float(np.mean(values[("frozen_bert_input", "auc")]))
        methods = [
            m for m in CL_METHODS_ORD
            if (m, "acc") in values and (m, "auc") in values
        ]
        acc_vals = [float(np.mean(values[(m, "acc")])) for m in methods]
        auc_vals = [float(np.mean(values[(m, "auc")])) for m in methods]
        split = "test"
    else:
        L = np.load("results_leakage.npy", allow_pickle=True).item()
        base = L["baseline"]
        per_m = L["methods"]
        split = L.get("split", "val")
        base_acc = float(np.mean([v["acc"] for v in base.values()]))
        base_auc = float(np.mean([v["auc"] for v in base.values()]))
        methods = [m for m in CL_METHODS_ORD if m in per_m]
        acc_vals = [float(np.mean([v["acc"] for v in per_m[m].values()])) for m in methods]
        auc_vals = [float(np.mean([v["auc"] for v in per_m[m].values()])) for m in methods]

    x       = np.arange(len(methods))
    labels  = [DISPLAY.get(m, m) for m in methods]
    cols    = [COLORS[m] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.2), layout="constrained")
    for ax, vals, base_v, ylabel, chance in [
        (axes[0], acc_vals, base_acc,
         "Gender probe accuracy\n(lower = less leakage; 0.5 = fair ideal)", 0.5),
        (axes[1], auc_vals, base_auc,
         "Gender probe AUC\n(lower = less leakage; 0.5 = fair ideal)", 0.5),
    ]:
        ax.set_axisbelow(True)
        ax.grid(axis="y", **GRID_KW)
        # Leakage is a fairness quantity: lower (closer to the 0.5 ideal) = fairer
        # = more blue. Anchor the ramp at the fair ideal so distance-from-fair reads.
        vmax = max(max(vals), base_v)
        bcols = value_colors(vals, "gap", vmin=chance, vmax=vmax)
        bars = ax.bar(x, vals, color=bcols, width=BAR_WIDTH, zorder=2,
                      edgecolor=BAR_EDGE, linewidth=BAR_EDGE_W)
        mark_our_bars(bars, methods)
        ax.axhline(chance, color="0.4", lw=0.9, ls=":", zorder=1,
                   label="chance (fair ideal)")
        ax.axhline(base_v, color="#9c755f", lw=1.1, ls="--", zorder=1,
                   label="frozen-BERT input (pre-training)")
        lo = min(chance, min(vals), base_v)
        hi = max(max(vals), base_v)
        ax.set_ylim(lo - 0.04 * (hi - lo + 1e-6), hi + 0.08 * (hi - lo + 1e-6))
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")

    fig.suptitle(f"Gender leakage in learned representations ({split} set; "
                 f"lower = fairer)", fontsize=11.0)
    out = os.path.join(OUT_DIR, "leakage.pdf")
    fig.savefig(out)
    print(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Projection activation diagnostics (mechanism; appendix)
# ══════════════════════════════════════════════════════════════════════════════

def plot_projection_diagnostics():
    """Per-task activation of the disparity-safety projection for FAIR-GEM and
    Fair-DER++ (the only two methods that run it).

    Left:  fraction of optimisation steps on which the safety constraint fires
           (g_safe . g_task < eps_t). FAIR-GEM also runs the A-GEM memory
           constraint, shown as a hatched overlay.
    Right: the dynamic margin eps_t (solid) tracks the inter-group disparity
           (DRO) loss L_worst - L_best (dashed) — confirming the margin is
           adaptive rather than fixed.

    Reads results_proj_stats.npy (written during training). Skips if absent.
    Dataset-agnostic: works for BIOS (per-task firing rates).
    """
    audit_path = os.path.join("..", "..", "analysis", "results", "mechanism_diagnostics.csv")
    audit_rows = []
    if os.path.exists(audit_path):
        with open(audit_path, newline="", encoding="utf-8") as fh:
            audit_rows = [
                row for row in csv.DictReader(fh)
                if row.get("benchmark") == "bios"
            ]
    if not audit_rows and not os.path.exists("results_proj_stats.npy"):
        print("Skipping projection diagnostics - results_proj_stats.npy not found.")
        return

    if audit_rows:
        grouped = {}
        for row in audit_rows:
            key = (row["method"], int(row["task"]))
            grouped.setdefault(key, []).append(row)
        P = {}
        for (method, task), rows in grouped.items():
            P.setdefault(method, {})[task] = {}
            for key in ("safety_proj_rate", "mem_proj_rate", "mean_eps_t", "mean_dro_loss"):
                vals = [float(row[key]) for row in rows if row.get(key, "") != ""]
                if vals:
                    P[method][task][key] = float(np.mean(vals))
    else:
        P = np.load("results_proj_stats.npy", allow_pickle=True).item()
    proj_methods = [m for m in ("fair_gem", "fair_derpp") if m in P]
    if not proj_methods:
        print("Skipping projection diagnostics - no FAIR-GEM/Fair-DER++ stats.")
        return

    tasks = sorted(P[proj_methods[0]].keys())
    xt    = np.arange(len(tasks))

    def series(m, key):
        return np.array([float(P[m][t].get(key, np.nan)) for t in tasks])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(FULL_W, 3.1),
                                   constrained_layout=True)

    # Left: safety (and, for FAIR-GEM, memory) projection firing rate
    nb = len(proj_methods)
    bw = 0.8 / nb
    for j, m in enumerate(proj_methods):
        off = (j - (nb - 1) / 2) * bw
        axL.bar(xt + off, 100 * series(m, "safety_proj_rate"), width=bw,
                color=COLORS[m], edgecolor=OURS_EDGE, linewidth=0.6,
                label=f"{DISPLAY.get(m, m)} — safety")
        if np.any(np.isfinite(series(m, "mem_proj_rate"))) and m == "fair_gem":
            axL.bar(xt + off, 100 * series(m, "mem_proj_rate"), width=bw,
                    facecolor="none", edgecolor=OURS_EDGE, linewidth=0.6,
                    hatch="////", label=f"{DISPLAY.get(m, m)} — memory")
    axL.set_xticks(xt); axL.set_xticklabels([f"T{t}" for t in tasks])
    axL.set_xlabel("Training task")
    axL.set_ylabel("Projection firing rate (%)")
    axL.set_title("(a) How often the constraint fires")
    axL.grid(axis="y", **GRID_KW); axL.set_axisbelow(True)
    axL.legend(fontsize=6.5, loc="best")

    # Right: dynamic margin eps_t (solid) vs DRO disparity loss (dashed)
    axR2 = axR.twinx()
    for m in proj_methods:
        axR.plot(xt, series(m, "mean_eps_t"), color=COLORS[m], lw=1.8, ls="-",
                 marker="o", ms=3, label=f"{DISPLAY.get(m, m)} — $\\varepsilon_t$")
        axR2.plot(xt, series(m, "mean_dro_loss"), color=COLORS[m], lw=1.4, ls="--",
                  marker="^", ms=3, alpha=0.8)
    axR.set_xticks(xt); axR.set_xticklabels([f"T{t}" for t in tasks])
    axR.set_xlabel("Training task")
    axR.set_ylabel(r"Dynamic margin $\varepsilon_t$ (solid)")
    axR2.set_ylabel(r"DRO disparity loss (dashed)")
    axR.set_title("(b) Margin tracks group disparity")
    axR.grid(axis="y", **GRID_KW); axR.set_axisbelow(True)
    axR.legend(fontsize=6.5, loc="upper left")

    fig.suptitle("Disparity-safety projection diagnostics (BIOS)", fontsize=11.0)
    out = os.path.join(OUT_DIR, "projection_diagnostics.pdf")
    fig.savefig(out)
    print(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Pareto frontier (balanced accuracy vs. fairness gap)
# ══════════════════════════════════════════════════════════════════════════════

def plot_pareto():
    """Pareto frontier: balanced accuracy vs. fairness gap, one panel per metric.

    Uses the shared renderer (draw_pareto_panel / pareto_legend) at
    single-column figsize (COL_W × PARETO_HEIGHT_COL); the oracle is shown but
    excluded from the frontier. Fairness axes: accuracy parity
    (PRIMARY — the gap companion of worst-group balanced accuracy, the quantity the
    disparity-safety gradient targets) and the TPR gap. EO is omitted because on
    25-class BIOS the FPR term is diluted and EO collapses onto TPR.

    Reads the module-level *_test dicts, so it follows the same
    --test / --retune / --retune-fair source selection as the rest of the script
    and writes pareto_frontier_<source>.pdf (e.g. pareto_frontier_retune.pdf).
    """
    methods = [m for m in ALL_METHODS_ORD if m in bal_acc_test and m in eo_test]

    # Accuracy parity (primary) then TPR.
    fair_metrics = []
    if acc_parity_test is not None:
        fair_metrics.append(("Acc. parity", acc_parity_test))
    if tpr_test is not None:
        fair_metrics.append(("TPR gap", tpr_test))
    if not fair_metrics:
        print("Skipping pareto - no accuracy-parity / TPR metrics available.")
        return

    n = len(fair_metrics)
    fig, axes = plt.subplots(1, n, figsize=(COL_W, PARETO_HEIGHT_COL),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, (short, fair_dict) in zip(axes, fair_metrics):
        acc_vals  = [_final_avg_test(bal_acc_test[m]) for m in methods]
        # Accuracy-weighted fairness gap: forgotten tasks (acc≈0) get ≈0 weight so
        # a collapsed prediction can't read as spuriously fair.
        fair_vals = [_acc_weighted_final(bal_acc_test[m], fair_dict[m]) for m in methods]
        # Shared renderer: gridless panel, dashed frontier (oracle excluded),
        # 'ideal' marker, standardized point markers/sizes.
        draw_pareto_panel(ax, fair_vals, acc_vals, methods,
                          f"{short} $\\downarrow$", show_ylabel=(ax is axes[0]))

    pareto_legend(fig, methods)   # standardized 2-row legend
    out = os.path.join(OUT_DIR, f"pareto_frontier_{_FINAL_SRC}.pdf")
    fig.savefig(out)   # exact figsize (global savefig.bbox=None)
    plt.close(fig)
    print(f"Saved {out}")


# Run all
if __name__ == "__main__":
    plot_summary()
    plot_heatmaps()
    plot_trajectories()
    plot_pareto()
    plot_head_vs_probe()
    plot_leakage()
    plot_projection_diagnostics()
    print("Done. Figures written to figures/")
