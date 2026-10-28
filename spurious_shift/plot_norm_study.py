"""
plot_norm_study.py — paper artifacts for the normalisation ablation.

Reads norm_study.csv (written by analysis/norm_study.py) and writes:

    figures/norm_study.pdf     three panels: BalAcc, accuracy-parity gap, F-BWT
    figures/norm_study.tex     booktabs table, ready to \\input
    figures/norm_study_hparams.tex   the tuned margin per condition

The disparity safety gradient is computed from a forward pass whose
normalisation we choose. BatchNorm in train mode normalises
by statistics pooled over the batch, so gradients couple samples across
demographic groups and the safety gradient can act through that coupling.
LayerNorm normalises per sample and has no such channel, and no running state.
Each condition is tuned on its own terms before being evaluated, because the
margin `eps_t = min(alpha, beta * L_disp)` is calibrated against the scale of
the disparity loss — which is exactly what the normalisation choice changes.

The LayerNorm column is the one that bears on scope: the Bias-in-Bios encoder is
BERT, which is LayerNorm throughout.

Usage (from spurious_shift/results):
    python ../plot_norm_study.py [--csv norm_study.csv]
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_style as S

# Display order and human labels. Kept explicit so the table reads the same way
# every time regardless of dict ordering in the CSV.
CONDITIONS = [
    ("bn-legacy", "BatchNorm, running stats updated"),
    ("bn-freeze", "BatchNorm, running stats frozen"),
    ("bn-eval",   "BatchNorm, eval-mode aux passes"),
    ("layernorm", "LayerNorm"),
]
SHORT = {"bn-legacy": "BN\nupdated", "bn-freeze": "BN\nfrozen",
         "bn-eval": "BN\neval", "layernorm": "Layer\nNorm"}

# All six methods are shown: LayerNorm also helps FaIRL substantially, so the
# LayerNorm result means "the mechanism does not depend on BatchNorm", not
# "only the fairness-aware methods benefit".
METHODS = ["agem", "der_pp", "group_dro", "fairl", "fair_gem", "fair_derpp"]
OURS = ["fair_gem", "fair_derpp"]

# (csv key, axis label, higher-is-better)
PANELS = [
    ("BalAcc",    r"balanced accuracy $\uparrow$", True),
    ("AccParity", f"accuracy-parity gap {S.D_AP} " + r"$\downarrow$", False),
    ("F-BWT",     r"fairness BWT $\uparrow$", True),
]


def load(path):
    rows = defaultdict(dict)
    hparams, n_seeds = {}, 0
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[r["condition"]][r["method"]] = r
            n_seeds = int(r["n_seeds"])
            if r.get("FAIRGEM_ALPHA"):
                hparams[r["condition"]] = {
                    k: r[k] for k in ("FAIRGEM_ALPHA", "FAIRGEM_BETA",
                                      "FAIRDERPP_EPS_ALPHA", "FAIRDERPP_EPS_BETA")
                    if r.get(k)}
    return rows, hparams, n_seeds


def figure(rows, hparams, n_seeds, out):
    S.set_style()
    conds = [c for c, _ in CONDITIONS if c in rows]
    fig, axes = plt.subplots(1, len(PANELS),
                             figsize=(S.NEURIPS_TEXTWIDTH, 2.7))
    x = np.arange(len(conds))
    width = 0.86 / len(METHODS)

    for ax, (key, label, higher) in zip(axes, PANELS):
        for j, m in enumerate(METHODS):
            mu = [float(rows[c][m][f"{key}_mean"]) if m in rows[c] else np.nan
                  for c in conds]
            sd = [float(rows[c][m][f"{key}_std"]) if m in rows[c] else np.nan
                  for c in conds]
            ax.bar(x + j * width - 0.43 + width / 2, mu, width,
                   yerr=sd if n_seeds > 1 else None, capsize=2,
                   label=S.display_name(m),
                   color=S.METHOD_COLORS.get(m, "0.6"),
                   hatch="//" if m in OURS else None,
                   edgecolor="white", linewidth=0.5, zorder=3,
                   error_kw=dict(lw=0.8, ecolor="0.25", zorder=4))
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT.get(c, c) for c in conds], fontsize=6.5)
        ax.set_ylabel(label, fontsize=7.5)
        ax.grid(axis="y", **S.GRID_KW)
        ax.set_axisbelow(True)
        if key == "F-BWT":
            ax.axhline(0.0, color="0.35", lw=0.7, zorder=2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(METHODS),
               frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.08))
    seedtxt = f"{n_seeds} seed" + ("s" if n_seeds != 1 else "")
    # Caption says per-condition optima only if the recorded hyperparameters
    # differ across conditions (a --skip-tuning run writes the same config
    # values into every row).
    distinct = len({tuple(sorted(h.items())) for h in hparams.values()})
    tuned = ("each condition at its own tuned optimum" if distinct > 1
             else "all conditions at one fixed configuration")
    fig.suptitle(f"Normalisation ablation — {tuned} ({seedtxt})",
                 fontsize=8.5, y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def table(rows, n_seeds, out):
    def cell(r, key):
        if r is None:
            return "--"
        mu, sd = float(r[f"{key}_mean"]), float(r[f"{key}_std"])
        return f"{mu:.3f} \\pm {sd:.3f}" if n_seeds > 1 else f"{mu:.3f}"

    lines = [
        "% Generated by plot_norm_study.py — do not edit by hand.",
        "\\begin{tabular}{llccc}", "\\toprule",
        "Normalisation & Method & BalAcc $\\uparrow$ & "
        "$\\Delta_{\\mathrm{AP}}$ $\\downarrow$ & F-BWT $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for cid, label in CONDITIONS:
        if cid not in rows:
            continue
        for i, m in enumerate(OURS):
            r = rows[cid].get(m)
            first = label if i == 0 else ""
            lines.append(
                f"{first} & {S.display_name(m, mark=False)} & "
                f"${cell(r, 'BalAcc')}$ & ${cell(r, 'AccParity')}$ & "
                f"${cell(r, 'F-BWT')}$ \\\\")
        lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out}")


def hparam_table(hparams, out):
    if not hparams:
        return
    lines = [
        "% Generated by plot_norm_study.py — do not edit by hand.",
        "\\begin{tabular}{lcccc}", "\\toprule",
        "Normalisation & \\multicolumn{2}{c}{FAIR-GEM} & "
        "\\multicolumn{2}{c}{Fair-DER++} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        " & $\\alpha$ & $\\beta$ & $\\varepsilon_\\alpha$ & $\\varepsilon_\\beta$ \\\\",
        "\\midrule",
    ]
    for cid, label in CONDITIONS:
        p = hparams.get(cid)
        if not p:
            continue
        lines.append(
            f"{label} & {float(p['FAIRGEM_ALPHA']):.3f} & {float(p['FAIRGEM_BETA']):.3f}"
            f" & {float(p['FAIRDERPP_EPS_ALPHA']):.3f}"
            f" & {float(p['FAIRDERPP_EPS_BETA']):.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="norm_study.csv")
    args = ap.parse_args()
    if not os.path.exists(args.csv):
        raise SystemExit(f"{args.csv} not found — run analysis/norm_study.py first.")

    rows, hparams, n_seeds = load(args.csv)
    os.makedirs("figures", exist_ok=True)
    figure(rows, hparams, n_seeds, "figures/norm_study.pdf")
    table(rows, n_seeds, "figures/norm_study.tex")
    hparam_table(hparams, "figures/norm_study_hparams.tex")
    if n_seeds < 2:
        print("\nNOTE: single seed, so the table has no error bars.")


if __name__ == "__main__":
    main()
