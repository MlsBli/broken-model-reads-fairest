"""
plot_motivation.py — Motivation figure for FAIR-GEM / Fair-DER++.

Two panels share the paper's colour semantic (see plot_style.py):
  (a) Avg. balanced accuracy across all seen tasks — every method sits near the
      top, so accuracy alone cannot separate them.
  (b) Accuracy-parity gap on the current task at training time — fine-tune and
      the non-fair baselines blow up once the spurious shortcut dominates,
      while our methods stay fair.

The shaded band marks the tasks where the shortcut dominates the legitimate
signal.

Output: figures/motivation.pdf

Run from spurious_shift/results:
    python ../plot_motivation.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import plot_style as S

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)
S.set_style()


# ── Load ──────────────────────────────────────────────────────────────────────

def _remap(d):
    return d


_RESULTS_PATH = S.resolve_results("results_agg.npz", "results.npz", "motivation figure")
data    = np.load(_RESULTS_PATH, allow_pickle=True)
states  = data["states"].tolist()
N       = len(states)
bal_acc = _remap(data["bal_acc"].item())
eo      = _remap(data["eo"].item())

XLABS = [S.task_label(s) for s in states]
x     = np.arange(N)

# Methods to show, in the paper legend order.
METHODS = [m for m in ["joint_retrain", "finetune", "agem", "der_pp",
                       "group_dro", "fairl", "fsw", "fair_gem", "fair_derpp"]
           if m in bal_acc]


# ── Series ─────────────────────────────────────────────────────────────────────

def _row_avg(mat, t):
    vals = [mat[t, i] for i in range(t + 1) if np.isfinite(mat[t, i])]
    return float(np.mean(vals)) if vals else np.nan


# Panel (a): running average balanced accuracy across all seen tasks.
acc_series = {m: [_row_avg(bal_acc[m], t) for t in range(N)] for m in METHODS}
# Panel (b): accuracy-parity gap on the current task right after training (diagonal).
eo_series  = {m: [eo[m][t, t] for t in range(N)] for m in METHODS}

# +/-1 SD across seeds, each curve recomputed within each seed first. Panel (a)
# averages several cells per point, so a per-cell std would understate it.
_extractors = {
    "acc": lambda mat: np.array([_row_avg(mat, t) for t in range(N)]),
    "eo":  lambda mat: np.array([mat[t, t] for t in range(N)]),
}


def _band(key, method, which):
    per = S.per_seed_stat(data, key, method, _extractors[which])
    if per is None:
        return None
    return np.nanstd(np.asarray(per, dtype=float), axis=0, ddof=1)


acc_sd = {m: _band("bal_acc", m, "acc") for m in METHODS}
eo_sd  = {m: _band("eo", m, "eo") for m in METHODS}


# ── Plot ───────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(S.COL_W, 2.5),
                         layout="constrained")

panels = [
    (axes[0], acc_series, acc_sd, False,
     r"Avg. seen-task accuracy $\uparrow$",
     "(a) Accuracy across methods"),
    (axes[1], eo_series, eo_sd, True,
     f"Current-task {S.D_AP} $\\downarrow$",
     "(b) Current-task fairness"),
]

banded = False
for ax, series, sds, anchor_zero, ylabel, title in panels:
    ax.set_axisbelow(True)
    ax.grid(axis="y", **S.GRID_KW)

    for m in METHODS:
        style = S.line_style(m)
        ax.plot(x, series[m], label=S.display_name(m), **style)
        sd = sds.get(m)
        if sd is not None:
            mu = np.asarray(series[m], dtype=float)
            ax.fill_between(x, mu - sd, mu + sd, color=style.get("color", "0.4"),
                            alpha=S.BAND_ALPHA, lw=0, zorder=1)
            banded = True

    ax.set_xticks(x)
    ax.set_xticklabels(XLABS)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=6)
    ax.set_xlim(-0.4, N - 0.6)
    extents = [np.asarray(series[m], dtype=float) + sign * sds[m]
               for m in METHODS if sds.get(m) is not None for sign in (-1, 1)]
    ax.set_ylim(*S.tight_ylim(list(series.values()) + extents,
                              anchor_zero=anchor_zero))
    ax.tick_params(axis="x", length=0)
    S.annotate_shortcut_band(ax, states)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=3,
           frameon=True, edgecolor="0.80",
           title=(f"shaded: {S.spread_note(data)}" if banded else None),
           title_fontsize=6.5)

out = os.path.join(OUT_DIR, "motivation.pdf")
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
