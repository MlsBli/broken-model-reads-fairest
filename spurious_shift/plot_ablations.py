"""
plot_ablations.py — Ablation figures for FAIR-GEM / Fair-DER++.

Reads results_test.npz (full-method results) and results_ablations.npz
(ablation variants), writes to figures/ablations/:
  ablations_fair_gem.pdf
  ablations_fair_derpp.pdf

Each figure has two panels: balanced accuracy (↑) and accuracy-parity gap (↓), comparing
the full method against its ablation variants.

Run from spurious_shift/results:
    python ../plot_ablations.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import plot_style as S

OUT_DIR = os.path.join("figures", "ablations")
os.makedirs(OUT_DIR, exist_ok=True)
S.set_style()


# ── Load ──────────────────────────────────────────────────────────────────────

def _remap(d):
    return d


_TEST_PATH = ("results_test_agg.npz" if os.path.exists("results_test_agg.npz")
              else "results_test.npz")
_ABLATION_PATH = S.resolve_results("results_ablations_agg.npz",
                                   "results_ablations.npz", "ablation figure")
test_data = np.load(_TEST_PATH, allow_pickle=True)
ablation_data = np.load(_ABLATION_PATH, allow_pickle=True)

test_bal_acc = _remap(test_data["bal_acc"].item())
test_eo      = _remap(test_data["eo"].item())
abl_bal_acc  = ablation_data["bal_acc"].item()
abl_eo       = ablation_data["eo"].item()

all_bal_acc = {**test_bal_acc, **abl_bal_acc}
all_eo      = {**test_eo,      **abl_eo}

N_TASKS = next(iter(all_bal_acc.values())).shape[0]


def _final_mean(mat):
    return float(np.nanmean(mat[N_TASKS - 1, :N_TASKS]))


# Each bar's whisker is the final-row mean recomputed inside every seed, then
# the sample SD across seeds — identical to the _std columns of
# results_ablations_summary.csv. The full-method bar comes from the test-split
# aggregate and the variants from the ablation aggregate, so the spread is
# looked up in whichever file supplied that bar.
def _err(variant, key):
    for source in (ablation_data, test_data):
        if S.per_seed(source, key, variant) is not None:
            return S.seed_sd_of(source, key, variant, _final_mean)
    return float("nan")


# ── Ablation group definitions ─────────────────────────────────────────────────

FAIR_GEM_VARIANTS = [
    "fair_gem",
    "fair_gem_static_eps",
    "fair_gem_avg_grad",
    "fair_gem_topk_dro",
    "fair_gem_uniform_memory",
    "fair_gem_alpha_low",
    "fair_gem_alpha_high",
    "fair_gem_beta_low",
    "fair_gem_beta_high",
]

FAIR_DERPP_VARIANTS = [
    "fair_derpp",
    "fair_derpp_static_eps",
    "fair_derpp_avg_grad",
    "fair_derpp_uniform_memory",
    "fair_derpp_alpha_low",
    "fair_derpp_alpha_high",
    "fair_derpp_beta_low",
    "fair_derpp_beta_high",
]

# X-axis tick labels (vertical, so length is unconstrained).
_LABELS = {
    "fair_gem":              "FAIR-GEM",
    "fair_gem_static_eps":   "Static $\\varepsilon$",
    "fair_gem_avg_grad":     "Avg grad",
    "fair_gem_topk_dro":     "Top-$k$ DRO",
    "fair_gem_uniform_memory": "Uniform memory",
    "fair_gem_alpha_low":     "$0.75\\alpha$",
    "fair_gem_alpha_high":    "$1.25\\alpha$",
    "fair_gem_beta_low":      "$0.75\\beta$",
    "fair_gem_beta_high":     "$1.25\\beta$",
    "fair_derpp":            "Fair-DER++",
    "fair_derpp_static_eps": "Static $\\varepsilon$",
    "fair_derpp_avg_grad":   "Avg grad",
    "fair_derpp_uniform_memory": "Uniform memory",
    # Fair-DER++ scales eps_alpha/eps_beta here, and separately carries an
    # unrelated alpha/beta (the DER++ logit-replay weights, 1.046 / 0.260). A
    # bare "0.75a" bar reads as the replay weight, so name the eps parameters.
    "fair_derpp_alpha_low":   "$0.75\\varepsilon_\\alpha$",
    "fair_derpp_alpha_high":  "$1.25\\varepsilon_\\alpha$",
    "fair_derpp_beta_low":    "$0.75\\varepsilon_\\beta$",
    "fair_derpp_beta_high":   "$1.25\\varepsilon_\\beta$",
}

# Colors: full method uses its canonical METHOD_COLORS entry; ablations get a
# muted version of the same hue so the full method stands out clearly.
_ABL_COLOR = "0.68"   # neutral grey for ablation bars


def _bar_color(variant):
    if variant in ("fair_gem", "fair_derpp"):
        return S.METHOD_COLORS.get(variant, "0.4")
    return _ABL_COLOR


# ── Plot ───────────────────────────────────────────────────────────────────────

def _plot_ablation_group(variants, title, out_path):
    missing = [v for v in variants if v not in all_bal_acc]
    if missing:
        print(f"WARNING: missing results for {missing} - skipping {title}")
        return

    ba_vals = [_final_mean(all_bal_acc[v]) for v in variants]
    eo_vals = [_final_mean(all_eo[v])      for v in variants]
    ba_errs = [_err(v, "bal_acc")          for v in variants]
    eo_errs = [_err(v, "eo")               for v in variants]
    if not any(np.isfinite(e) for e in ba_errs):
        ba_errs = eo_errs = None
    labels  = [_LABELS.get(v, v)           for v in variants]
    colors  = [_bar_color(v)               for v in variants]
    x       = np.arange(len(variants))

    UP, DN = r"$\uparrow$", r"$\downarrow$"
    panel_specs = [
        (ba_vals, ba_errs, f"Balanced accuracy {UP}", "acc",  True),
        (eo_vals, eo_errs, f"accuracy-parity gap {S.D_AP} {DN}",  "gap",  False),
    ]

    # Full-width figure so both panels sit comfortably; height leaves room for
    # vertical tick labels (constrained_layout handles the bottom margin).
    fig, axes = plt.subplots(1, 2, figsize=(S.FULL_W, 2.9),
                             layout="constrained")
    fig.get_layout_engine().set(w_pad=0.10, wspace=0.18)

    for ax, (vals, errs, ylabel, channel, higher_better) in zip(axes, panel_specs):
        ax.set_axisbelow(True)
        ax.grid(axis="y", **S.GRID_KW)

        bars = ax.bar(x, vals, color=colors,
                      width=S.BAR_WIDTH, zorder=2,
                      edgecolor=S.BAR_EDGE, linewidth=S.BAR_EDGE_W,
                      yerr=errs, error_kw=dict(zorder=5, **S.ERRORBAR_KW))

        # Apply the shared "our methods" hatch/edge treatment to the full method.
        S.mark_our_bars(bars, variants)

        # Dashed reference line at the full-method value (first bar).
        full_val = vals[0]
        if np.isfinite(full_val):
            ax.axhline(full_val, color=S.METHOD_COLORS.get(variants[0], "0.4"),
                       lw=0.9, ls=(0, (4, 3)), zorder=1, alpha=0.75)

        # Y-axis from 0; 20% headroom above the tallest whisker for value labels.
        tops = [v + (e if errs is not None and np.isfinite(e) else 0.0)
                for v, e in zip(vals, errs if errs is not None else [0.0] * len(vals))
                if np.isfinite(v)]
        yhi = max(tops) * 1.20
        ax.set_ylim(0, yhi)

        # Value labels, lifted clear of the error-bar cap.
        for i, (bar, val) in enumerate(zip(bars, vals)):
            if not np.isfinite(val):
                continue
            e = errs[i] if errs is not None else 0.0
            e = e if np.isfinite(e) else 0.0
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + e + 0.015 * yhi,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=S.VALUE_LABEL_SIZE - 0.5)

        ax.set_xticks(x)
        # Vertical labels prevent any overlap regardless of label length.
        ax.set_xticklabels(labels, rotation=90, ha="center", va="top")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.65, len(variants) - 0.35)

    # Legend: full method (hatched) vs ablations.
    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(facecolor=S.METHOD_COLORS.get(variants[0], "0.4"),
                       edgecolor=S.OURS_EDGE, hatch=S.HATCH_OURS,
                       label=f"{S.STAR} full method (hatched)"),
        mpatches.Patch(facecolor=_ABL_COLOR, edgecolor=S.BAR_EDGE,
                       label="ablation variant"),
    ]
    if ba_errs is not None:
        legend_elements.append(
            plt.Line2D([0], [0], color=S.ERRORBAR_KW["ecolor"],
                       lw=S.ERRORBAR_KW["elinewidth"], marker="|", markersize=5,
                       label=S.seed_note(ablation_data)))
    fig.legend(handles=legend_elements, loc="outside upper center", ncol=2,
               frameon=True, edgecolor="0.80", fontsize=7.5,
               handlelength=2.2, columnspacing=2.5)

    out = out_path
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    _plot_ablation_group(
        FAIR_GEM_VARIANTS,
        "FAIR-GEM Ablations",
        os.path.join(OUT_DIR, "ablations_fair_gem.pdf"),
    )
    _plot_ablation_group(
        FAIR_DERPP_VARIANTS,
        "Fair-DER++ Ablations",
        os.path.join(OUT_DIR, "ablations_fair_derpp.pdf"),
    )
    print("Done. Figures written to figures/ablations/")
