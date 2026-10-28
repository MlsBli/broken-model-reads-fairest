"""
plot_results.py — figures for FAIR-GEM / Fair-DER++ on spurious_shift.

Reads results.npz and results_test.npz, writes:
  summary_accuracy.pdf
  summary_fairness.pdf
  summary_main.pdf
  forgetting_heatmaps.pdf
  eo_dp_trajectory.pdf
  pareto_frontier.pdf
  tradeoff_strength.pdf

Run from spurious_shift/results:
    python ../plot_results.py
"""

import csv
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import plot_style as S

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)
S.set_style()


# Load

def _remap(d):
    return d


_TRAIN_PATH = S.resolve_results("results_agg.npz", "results.npz", "trajectory/BWT panels")
_TEST_PATH = ("results_test_agg.npz" if os.path.exists("results_test_agg.npz")
              else "results_test.npz")
data = np.load(_TRAIN_PATH, allow_pickle=True)
data_test = np.load(_TEST_PATH, allow_pickle=True)

states = data["states"].tolist()
N = len(states)

bal_acc = _remap(data["bal_acc"].item())
eo = _remap(data["eo"].item())
dp = _remap(data["dp"].item()) if "dp" in data else None

bal_acc_test = _remap(data_test["bal_acc"].item())
eo_test = _remap(data_test["eo"].item())
dp_test = _remap(data_test["dp"].item()) if "dp" in data_test else None

XLABS = [S.task_label(s) for s in states]
TICKS_SHORT = [S.task_tick(s) for s in states]


# Methods absent from a given run are filtered out below, so order is all that matters.
ALL_METHODS_ORD = [
    "joint_retrain",
    "finetune", "agem", "der_pp", "group_dro", "fairl", "fsw",
    "fair_gem", "fair_derpp",
]
CL_METHODS_ORD = [
    "joint_retrain", "finetune", "agem", "der_pp",
    "group_dro", "fairl", "fsw", "fair_gem", "fair_derpp",
]

SUMMARY_METHODS = [m for m in ALL_METHODS_ORD if m in bal_acc_test]
CL_METHODS = [m for m in CL_METHODS_ORD  if m in bal_acc]



def _final_avg_test(mat):
    return float(np.nanmean(mat[N - 1, :]))


def _bwt(bal_acc_mat):
    """Accuracy backward transfer"""
    vals = [bal_acc_mat[N - 1, i] - bal_acc_mat[i, i]
            for i in range(N - 1)
            if np.isfinite(bal_acc_mat[N - 1, i]) and np.isfinite(bal_acc_mat[i, i])]
    return float(np.mean(vals)) if vals else float("nan")


def _f_bwt(eo_mat):
    """Fairness backward transfer"""
    vals = [eo_mat[i, i] - eo_mat[N - 1, i]
            for i in range(N - 1)
            if np.isfinite(eo_mat[N - 1, i]) and np.isfinite(eo_mat[i, i])]
    return float(np.mean(vals)) if vals else float("nan")


def _acc_weighted_gap(acc_mat, fair_mat):
    """Accuracy-weighted final fairness gap"""
    accs = np.array([acc_mat[N - 1, e]  for e in range(N)])
    gaps = np.array([fair_mat[N - 1, e] for e in range(N)])
    mask = np.isfinite(accs) & np.isfinite(gaps)
    if mask.sum() == 0:
        return float("nan")
    w = np.clip(accs[mask], 0, None)
    if w.sum() < 1e-8:
        return float(np.mean(gaps[mask]))
    return float(np.average(gaps[mask], weights=w))


def _errs(source, key, methods, fn):
    """Per-method +/-1 SD across seeds of `fn`, or None if the file cannot say.

    None (rather than zeros) when the aggregate carries no per-seed stacks,
    since a single-seed run has no measured spread.
    """
    vals = [S.seed_sd_of(source, key, m, fn) for m in methods]
    return vals if any(np.isfinite(v) for v in vals) else None


def _pareto_frontier(points):
    """Indices of the accuracy-fairness Pareto frontier"""
    order = sorted(range(len(points)), key=lambda i: points[i][0])
    frontier, best_acc = [], -np.inf
    for i in order:
        if points[i][1] >= best_acc:
            frontier.append(i)
            best_acc = points[i][1]
    return frontier


# Figure 1 — Summary comparison bar charts

def plot_summary():
    methods = SUMMARY_METHODS
    x = np.arange(len(methods))
    labels = [S.display_name(m) for m in methods]

    bal_vals = [_final_avg_test(bal_acc_test[m]) for m in methods]
    eo_vals = [_final_avg_test(eo_test[m])      for m in methods]
    dp_vals = ([_final_avg_test(dp_test[m]) for m in methods]
                 if dp_test is not None else None)
    bwt_vals = [_bwt(bal_acc[m])  for m in methods]
    fbwt_vals = [_f_bwt(eo[m])     for m in methods]

    # Each statistic is recomputed inside every seed and the spread taken across
    # seeds — the same route aggregate.py uses for the CSV, so these bars carry
    # exactly the _std columns of results_summary.csv. BWT needs the diagonal, so
    # it comes from the train-time file; the level metrics from the test file.
    bal_err = _errs(data_test, "bal_acc", methods, _final_avg_test)
    eo_err = _errs(data_test, "eo", methods, _final_avg_test)
    bwt_err = _errs(data, "bal_acc", methods, _bwt)
    fbwt_err = _errs(data, "eo", methods, _f_bwt)

    UP, DN = r"$\uparrow$", r"$\downarrow$"
    acc_specs = [
        (bal_vals, bal_err, f"Balanced accuracy {UP}",  "acc",  True,  False),
        (bwt_vals, bwt_err, f"Accuracy BWT {UP}",        "acc",  True,  True),
    ]
    # Two panels only (accuracy-parity gap + Fairness-BWT) so the figure stays
    # legible at single-column width; the DP gap is reported in the results
    # table.
    fair_specs = [
        (eo_vals, eo_err, f"accuracy-parity gap {S.D_AP} {DN}", "fair", False, False, True),
        (fbwt_vals, fbwt_err, f"Fairness BWT {UP}",        "fair", True,  True,  False),
    ]

    oracle_idx = methods.index(S.ORACLE) if S.ORACLE in methods else None

    legend_elements = [
        mpatches.Patch(facecolor="0.80", edgecolor=S.OURS_EDGE, hatch=S.HATCH_OURS,
                       label=f"{S.STAR} ours (hatched)"),
        plt.Line2D([0], [0], color="0.45", lw=1.1, ls=(0, (4, 3)),
                   label=f"{S.DAGGER} joint-training level"),
    ]
    # Name the interval (SD across seeds) on the figure.
    if bal_err is not None:
        legend_elements.append(
            plt.Line2D([0], [0], color=S.ERRORBAR_KW["ecolor"],
                       lw=S.ERRORBAR_KW["elinewidth"], marker="|",
                       markersize=5, label=S.seed_note(data_test)))

    def _draw_panel(ax, vals, errs, ylabel, channel, higher_better, zero_line,
                    zero_bottom=False):
        ax.set_axisbelow(True)
        ax.grid(axis="y", **S.GRID_KW)

        colors = S.value_colors(vals, channel=channel, higher_better=higher_better)
        bars = ax.bar(x, vals, color=colors, width=S.BAR_WIDTH, zorder=2,
                      edgecolor=S.BAR_EDGE, linewidth=S.BAR_EDGE_W,
                      yerr=errs, error_kw=dict(zorder=5, **S.ERRORBAR_KW))

        # Hatch our methods so they read even in greyscale (shared treatment).
        S.mark_our_bars(bars, methods)

        ax.set_ylim(*S.tight_ylim(vals, anchor_zero=zero_line, err=errs))
        if zero_bottom:
            ax.set_ylim(bottom=0)

        if zero_line:
            ax.axhline(0, color="0.45", lw=0.8, zorder=1)
        if oracle_idx is not None and np.isfinite(vals[oracle_idx]):
            ax.axhline(vals[oracle_idx], color="0.45", lw=0.9, ls=(0, (4, 3)),
                       zorder=1, alpha=0.8)

        # Tag oracle / our methods above their bars.
        ylo, yhi = ax.get_ylim()
        span = yhi - ylo
        for i, m in enumerate(methods):
            v = vals[i]
            if not np.isfinite(v):
                continue
            tag = S.DAGGER if m == S.ORACLE else (S.STAR if m in S.OURS else None)
            if tag:
                # Anchor past the error-bar cap, not the bar top, or the tag
                # lands on top of the whisker.
                e = errs[i] if errs is not None else 0.0
                e = e if np.isfinite(e) else 0.0
                anchor = v + e if v >= 0 else v - e
                off = 0.025 * span if v >= 0 else -0.06 * span
                ax.text(i, anchor + off, tag, ha="center",
                        va="bottom" if v >= 0 else "top",
                        fontsize=8, color="0.15", zorder=4)

        ax.set_xticks(x)
        # Single-column figure: vertical labels so the 8 method names don't
        # overlap in the narrow (2-panel) column.
        ax.set_xticklabels(labels, rotation=90, ha="center")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.65, len(methods) - 0.35)
        ax.tick_params(axis="x", length=0)

    def _make_figure(specs, fname):
        n = len(specs)
        fig, axes = plt.subplots(1, n, figsize=(S.COL_W, 2.48),
                                 layout="constrained")
        fig.get_layout_engine().set(w_pad=0.08, wspace=0.14)
        for ax, spec in zip(np.atleast_1d(axes), specs):
            _draw_panel(ax, *spec)
        fig.legend(handles=legend_elements, loc="outside upper center", ncol=2,
                   frameon=True, edgecolor="0.80", fontsize=9,
                   handlelength=2.2, columnspacing=2.5)
        out = os.path.join(OUT_DIR, fname)
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")

    # Two separate figures so each panel keeps a comfortable size.
    _make_figure(acc_specs,  "summary_accuracy.pdf")
    _make_figure(fair_specs, "summary_fairness.pdf")

    # Combined four-panel version: one legend and one set of method labels
    # for a single-column layout.
    specs = acc_specs + fair_specs
    fig, axes = plt.subplots(2, 2, figsize=(S.COL_W, 4.05),
                             layout="constrained")
    fig.get_layout_engine().set(w_pad=0.08, h_pad=0.06,
                                wspace=0.14, hspace=0.08)
    for idx, (ax, spec) in enumerate(zip(axes.flat, specs)):
        _draw_panel(ax, *spec)
        if idx < 2:
            ax.tick_params(axis="x", labelbottom=False)
    fig.legend(handles=legend_elements, loc="outside upper center", ncol=2,
               frameon=True, edgecolor="0.80", fontsize=8.5,
               handlelength=2.2, columnspacing=2.2)
    out = os.path.join(OUT_DIR, "summary_main.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# Figure 2 — Forgetting heat-maps

def plot_heatmaps():
    methods = [m for m in CL_METHODS if m in bal_acc]
    ncols = min(3, len(methods))
    nrows = (len(methods) + ncols - 1) // ncols

    # Three panels per row keep cell values and seed spreads legible at \linewidth.
    _panel = S.FULL_W / ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(S.FULL_W, _panel * nrows + 0.4 * nrows),
                             constrained_layout=True)
    axes_flat = np.atleast_1d(axes).flatten()

    all_vals = np.concatenate([
        [bal_acc[m][t, e] for t in range(N) for e in range(t + 1)]
        for m in methods])
    norm = S.heatmap_norm(all_vals)
    cmap = S.CMAP_ACC
    im_ref = None
    for ax, m in zip(axes_flat, methods):
        mat = np.full((N, N), np.nan)
        for t in range(N):
            for e in range(t + 1):
                mat[t, e] = bal_acc[m][t, e]

        im_ref = ax.imshow(mat, norm=norm, cmap=cmap, aspect="equal")

        # Cell means are the primary matrix diagnostic. Per-cell SD labels make
        # the 3x3 paper layout unreadable; uncertainty for each method remains in
        # the title as the SD of seed-level final-task averages.
        for t in range(N):
            for e in range(t + 1):
                v = mat[t, e]
                if np.isfinite(v):
                    tcol = "white" if norm(v) > 0.6 else "0.15"
                    ax.text(e, t, f"{v:.2f}", ha="center", va="center",
                            fontsize=8.0, color=tcol)

        ax.set_xticks(range(N)); ax.set_xticklabels(TICKS_SHORT)
        ax.set_yticks(range(N)); ax.set_yticklabels(TICKS_SHORT)
        ax.set_ylabel("after training on", fontsize=8.0)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)

        avg = float(np.nanmean(bal_acc[m][N - 1, :N]))
        avg_sd = S.seed_sd_of(data, "bal_acc", m, lambda mm: float(np.nanmean(mm[N - 1, :N])))
        title = (f"{S.display_name(m)} (avg {avg:.2f}$\\pm${avg_sd:.3f})"
                 if np.isfinite(avg_sd) else f"{S.display_name(m)} (avg {avg:.2f})")
        ax.set_title(title, fontsize=8.0)

    for ax in axes_flat[len(methods):]:
        ax.set_visible(False)

    if im_ref is not None:
        cbar = fig.colorbar(im_ref, ax=axes_flat[:len(methods)].tolist(),
                            fraction=0.020, pad=0.02)
        cbar.set_label("balanced accuracy  (greener = more accurate)", fontsize=8.0)
        cbar.outline.set_visible(False)

    title = "Per-task balanced accuracy after sequential training"
    fig.suptitle(title, fontsize=9.0)
    out = os.path.join(OUT_DIR, "forgetting_heatmaps.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# Figure 3 — Trajectories

def plot_trajectories():
    x = np.arange(N)
    shown = [m for m in CL_METHODS_ORD if m in eo and m in bal_acc]

    UP, DN = r"$\uparrow$", r"$\downarrow$"
    # (data_dict, metric_key, ylabel, title, anchor_zero, extractor)
    # metric_key names the per-seed stack so each curve's band is the spread of
    # that curve recomputed within each seed. Panel (b) averages several cells
    # per point, so its band cannot come from per-cell stds.
    col_specs = [
        (bal_acc, "bal_acc",
         f"Task-0 accuracy {UP}",
         "(a) Forgetting",
         False,
         lambda mat: np.array([mat[t, 0] for t in range(N)])),
        (bal_acc, "bal_acc",
         f"Avg. seen-task accuracy {UP}",
         "(b) Adaptability",
         False,
         lambda mat: np.array([np.nanmean(mat[t, :t + 1]) for t in range(N)])),
        (eo, "eo",
         f"Current-task {S.D_AP} {DN}",
         "(c) Fairness",
         True,
         lambda mat: np.array([mat[t, t] for t in range(N)])),
    ]

    fig, axes = plt.subplots(1, len(col_specs),
                             figsize=(S.FULL_W, 3.2),
                             constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.10, wspace=0.06)

    banded = False
    for ax, (metric, key, ylabel, title, anchor_zero, extractor) in zip(axes, col_specs):
        ax.set_axisbelow(True)
        ax.grid(axis="y", **S.GRID_KW)

        series, extents = {}, []
        for m in shown:
            series[m] = extractor(metric[m])
            style = S.line_style(m)
            ax.plot(x, series[m], label=S.display_name(m), **style)

            # +/-1 SD band: the curve recomputed inside each seed, then the
            # spread taken per task across seeds.
            per = S.per_seed_stat(data, key, m, extractor)
            if per is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    sd = np.nanstd(np.asarray(per, dtype=float), axis=0, ddof=1)
                lo, hi = series[m] - sd, series[m] + sd
                ax.fill_between(x, lo, hi, color=style.get("color", "0.4"),
                                alpha=S.BAND_ALPHA, lw=0, zorder=1)
                extents += [lo, hi]
                banded = True

        ax.set_ylim(*S.tight_ylim(list(series.values()) + extents,
                                  anchor_zero=anchor_zero))
        S.annotate_shortcut_band(ax, states)

        ax.set_xticks(x)
        ax.set_xticklabels(XLABS)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(-0.4, N - 0.6)
        ax.tick_params(axis="x", length=0)

    handles, labels = axes[0].get_legend_handles_labels()
    # The band note rides on the legend rather than a suptitle, which would sit
    # on top of the per-panel titles.
    fig.legend(handles, labels, loc="outside lower center",
               ncol=len(shown), frameon=True, edgecolor="0.80",
               title=(f"shaded: {S.spread_note(data)}" if banded else None),
               title_fontsize=6.5)

    out = os.path.join(OUT_DIR, "eo_dp_trajectory.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# Figure 4 — Pareto frontier

def plot_pareto():
    """Plots pareto frontier figure"""
    methods = [m for m in ALL_METHODS_ORD if m in bal_acc_test and m in eo_test]

    fair_metrics = [(f"accuracy-parity gap {S.D_AP}", eo_test)]
    if dp_test is not None:
        fair_metrics.append((f"demographic-parity gap {S.D_DP}", dp_test))

    n = len(fair_metrics)
    fig, axes = plt.subplots(1, n, figsize=(S.COL_W, S.PARETO_HEIGHT_COL),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, (fair_label, fair_dict) in zip(axes, fair_metrics):
        acc_vals = [_final_avg_test(bal_acc_test[m])        for m in methods]
        fair_vals = [_acc_weighted_gap(bal_acc_test[m], fair_dict[m]) for m in methods]
        fair_sym = S.D_AP if fair_dict is eo_test else S.D_DP

        acc_err = _errs(data_test, "bal_acc", methods, _final_avg_test)
        # The x statistic weights each task's gap by that task's accuracy, so it
        # is not linear in the matrix cells and has to be recomputed seed by
        # seed — it pairs the accuracy and gap matrices of the SAME run.
        fair_key = "eo" if fair_dict is eo_test else "dp"
        fair_err = []
        for m in methods:
            accs = S.per_seed(data_test, "bal_acc", m)
            gaps = S.per_seed(data_test, fair_key, m)
            fair_err.append(float("nan") if accs is None or gaps is None else
                            S.seed_sd([_acc_weighted_gap(a, g)
                                       for a, g in zip(accs, gaps)]))
        if not any(np.isfinite(v) for v in fair_err):
            fair_err = None

        # Shared renderer (also used for the Bias-in-Bios Pareto plots): grid,
        # frontier (oracle excluded), 'ideal', standardized markers.
        S.draw_pareto_panel(ax, fair_vals, acc_vals, methods,
                            f"{fair_sym} $\\downarrow$", show_ylabel=(ax is axes[0]),
                            fair_err=fair_err, acc_err=acc_err)

    S.pareto_legend(fig, methods, note=S.seed_note(data_test))
    out = os.path.join(OUT_DIR, "pareto_frontier.pdf")
    fig.savefig(out)   # exact figsize (global savefig.bbox=None)
    plt.close(fig)
    print(f"Saved {out}")


# Figure 5 — within-method fairness-control sweeps

def plot_tradeoff_strength():
    """Plot cross-seed accuracy/fairness curves over the control multiplier."""
    path = "tradeoff_curves_agg.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    methods = ["fair_derpp", "fair_gem", "fsw"]
    expected = set(methods)
    present = {row["method"] for row in rows}
    if present != expected:
        raise ValueError(f"{path}: expected methods {sorted(expected)}, got {sorted(present)}")

    fig, axes = plt.subplots(1, 2, figsize=(S.COL_W, 2.42),
                             constrained_layout=True)
    panels = [
        ("acc_parity", S.D_AP, "Accuracy parity"),
        ("eo", S.D_EO, "Equalized odds"),
    ]
    all_handles = []
    for ax, (metric, symbol, title) in zip(axes, panels):
        panel_x, panel_y, panel_xerr, panel_yerr = [], [], [], []
        for method in methods:
            arm = sorted(
                (row for row in rows if row["method"] == method),
                key=lambda row: float(row["scale"]),
            )
            scale = np.asarray([float(row["scale"]) for row in arm])
            x = np.asarray([float(row[f"{metric}_mean"]) for row in arm])
            y = np.asarray([float(row["bal_mean"]) for row in arm])
            xerr = np.asarray([float(row[f"{metric}_std"]) for row in arm])
            yerr = np.asarray([float(row["bal_std"]) for row in arm])
            style = S.line_style(method)
            color = style["color"]
            marker = style["marker"]
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none",
                        zorder=1, **S.ERRORBAR_KW)
            ax.plot(x, y, color=color, ls=style["ls"], lw=style["lw"],
                    zorder=2)
            sizes = 22 + 30 * scale / max(scale.max(), 1e-8)
            ax.scatter(x, y, s=sizes, marker=marker, color=color,
                       edgecolors=S.OURS_EDGE if method in S.OURS else "0.35",
                       linewidths=0.7, zorder=3)
            selected = np.isclose(scale, 1.0)
            ax.scatter(x[selected], y[selected], s=sizes[selected] + 34,
                       marker=marker, facecolors="none", edgecolors="black",
                       linewidths=1.0, zorder=4)
            panel_x.extend(x)
            panel_y.extend(y)
            panel_xerr.extend(xerr)
            panel_yerr.extend(yerr)
            if ax is axes[0]:
                all_handles.append(Line2D(
                    [0], [0], color=color, ls=style["ls"], lw=style["lw"],
                    marker=marker, markersize=5, label=S.display_name(method),
                ))

        xlo, xhi = S.tight_ylim(panel_x, margin=0.12, anchor_zero=True,
                                err=panel_xerr)
        ax.set_xlim(max(0.0, xlo), xhi)
        ax.set_ylim(*S.tight_ylim(panel_y, margin=0.12, err=panel_yerr))
        ax.set_xlabel(f"{symbol} gap $\\downarrow$")
        ax.set_title(title)
        ax.grid(False)
        ax.annotate("ideal", xy=(0.03, 0.97), xycoords="axes fraction",
                    ha="left", va="top", fontsize=7, style="italic", color="0.45")
    axes[0].set_ylabel(r"Balanced accuracy $\uparrow$")
    all_handles.extend([
        Line2D([0], [0], marker="o", ls="", markerfacecolor="0.45",
               markeredgecolor="black", markeredgewidth=1.0, markersize=6,
               label=r"ring: selected $k=1$"),
        Line2D([0], [0], marker="o", ls="", markerfacecolor="0.45",
               markeredgecolor="0.35", markersize=4,
               label=r"marker size: $k\uparrow$"),
    ])
    fig.legend(handles=all_handles, loc="outside lower center", ncol=5,
               frameon=True, edgecolor="0.80")
    out = os.path.join(OUT_DIR, "tradeoff_strength.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")



if __name__ == "__main__":
    plot_summary()
    plot_heatmaps()
    plot_trajectories()
    plot_pareto()
    plot_tradeoff_strength()
    print("Done. Figures written to figures/")
