"""
surrogate_alignment.py — Do the CE group-loss disparity surrogates track the
prediction-space fairness metrics?

Pools every (method, seed, train-task t, eval-task e) cell from
runs/seed*/results.npz under --results-dir and reports Spearman rho (with a
method-seed cluster-bootstrap CI) between each surrogate (loss_gap,
loss_gap_cond) and each
fairness metric present. Reports BOTH pooled and within-method correlations:
pooled rho can be manufactured entirely by between-method spread, so the
within-method numbers are the load-bearing ones.

Outputs: surrogate_alignment.csv and figures/surrogate_alignment.pdf in
--results-dir.

Usage: python analysis/surrogate_alignment.py --results-dir spurious_shift/results [--seeds 0 1 2 3 4]
"""

import argparse
import csv
import glob
import os

import numpy as np
from scipy.stats import spearmanr


PAPER_METHODS = {
    "joint_retrain": "Joint Retrain",
    "finetune": "Fine-tune",
    "agem": "A-GEM",
    "der_pp": "DER++",
    "derpp": "DER++",
    "group_dro": "GroupDRO",
    "fairl": "FaIRL-insp.",
    "fsw": "FSW",
    "fair_gem": "FAIR-GEM",
    "fair_derpp": "Fair-DER++",
}

PAPER_COLORS = {
    "joint_retrain": "#7F7F7F",
    "finetune": "#D1495B",
    "agem": "#6A4C93",
    "der_pp": "#E8833A",
    "derpp": "#E8833A",
    "group_dro": "#C44E9D",
    "fairl": "#8C6D4F",
    "fsw": "#0F8B8D",
    "fair_gem": "#2E8B57",
    "fair_derpp": "#16513A",
}

SURROGATES = ["loss_gap", "loss_gap_cond"]
METRICS = ["eo", "eo_true", "tpr", "acc_parity", "dp"]  # whichever exist
N_BOOT = 1000


def _load_cells(results_dir, seeds):
    """Load finite cells with parallel method and method-seed cluster labels."""
    files = sorted(glob.glob(os.path.join(results_dir, "runs", "seed*", "results.npz")))
    if seeds is not None:
        want = {f"seed{s}" for s in seeds}
        files = [f for f in files if os.path.basename(os.path.dirname(f)) in want]
    if not files:
        return _load_aggregate_cells(results_dir, seeds)
    first = np.load(files[0], allow_pickle=True)
    surro = [k for k in SURROGATES if k in first.files]
    mets = [k for k in METRICS if k in first.files]
    if not surro:
        raise SystemExit("results.npz has no surrogate matrices (loss_gap*) — "
                         "rerun training with the instrumented eval loop first.")
    cols = {k: [] for k in surro + mets}
    labels = []
    clusters = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        seed_label = os.path.basename(os.path.dirname(f))
        methods = [str(m) for m in d["methods"]]
        for m in methods:
            mats = {k: np.asarray(d[k].item()[m], dtype=float) for k in cols}
            mask = np.isfinite(next(iter(mats.values())))
            for k, M in mats.items():
                mask &= np.isfinite(M)
            idx = np.where(mask)
            n = len(idx[0])
            for k, M in mats.items():
                cols[k].extend(M[idx].tolist())
            labels.extend([m] * n)
            clusters.extend([f"{m}:{seed_label}"] * n)
    return ({k: np.asarray(v) for k, v in cols.items()}, np.asarray(labels),
            np.asarray(clusters), surro, mets)


def _load_aggregate_cells(results_dir, seeds):
    """Load exact per-seed matrices retained in the compact aggregate artifact."""
    path = os.path.join(results_dir, "results_agg.npz")
    if not os.path.exists(path):
        raise SystemExit(
            f"No runs/seed*/results.npz or results_agg.npz under {results_dir}"
        )
    data = np.load(path, allow_pickle=True)
    methods = [str(method) for method in data["methods"]]
    available = set(data.files)
    surro = [
        key for key in SURROGATES
        if any(f"{key}__{method}__per_seed" in available for method in methods)
    ]
    mets = [
        key for key in METRICS
        if any(f"{key}__{method}__per_seed" in available for method in methods)
    ]
    if not surro:
        raise SystemExit("results_agg.npz has no per-seed surrogate matrices")

    stored_seeds = [int(seed) for seed in data["seeds"]]
    seed_indices = range(len(stored_seeds))
    if seeds is not None:
        wanted = set(seeds)
        seed_indices = [i for i, seed in enumerate(stored_seeds) if seed in wanted]

    cols = {key: [] for key in surro + mets}
    labels = []
    clusters = []
    for seed_idx in seed_indices:
        seed_label = f"seed{stored_seeds[seed_idx]}"
        for method in methods:
            mats = {
                key: np.asarray(data[f"{key}__{method}__per_seed"][seed_idx], dtype=float)
                for key in cols
            }
            mask = np.isfinite(next(iter(mats.values())))
            for matrix in mats.values():
                mask &= np.isfinite(matrix)
            idx = np.where(mask)
            for key, matrix in mats.items():
                cols[key].extend(matrix[idx].tolist())
            labels.extend([method] * len(idx[0]))
            clusters.extend([f"{method}:{seed_label}"] * len(idx[0]))
    return {key: np.asarray(value) for key, value in cols.items()}, \
        np.asarray(labels), np.asarray(clusters), surro, mets


def _display_metric(metric, metrics):
    if metric == "eo":
        return "AP gap" if "eo_true" in metrics else "EO gap"
    return {
        "eo_true": "EO gap",
        "tpr": "TPR gap",
        "acc_parity": "AP gap",
        "dp": "DP gap",
    }.get(metric, metric)


def _display_surrogate(surrogate):
    return {
        "loss_gap": "Loss gap",
        "loss_gap_cond": "Label-conditioned loss gap",
    }.get(surrogate, surrogate)


def _rho_ci(x, y, clusters, rng):
    """Spearman rho with resampling at the method-seed cluster level."""
    rho = spearmanr(x, y).statistic
    unique = np.unique(clusters)
    boots = []
    for _ in range(N_BOOT):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(clusters == cluster)
                              for cluster in sampled])
        boots.append(spearmanr(x[idx], y[idx]).statistic)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return rho, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--methods", default=None,
                    help="comma-list; restrict to these methods (e.g. exclude "
                         "collapsed models whose degenerate EO dilutes the pool)")
    ap.add_argument("--suffix", default="", help="output filename suffix")
    args = ap.parse_args()

    cols, labels, clusters, surro, mets = _load_cells(args.results_dir, args.seeds)
    if args.methods:
        keep = np.isin(labels, args.methods.split(","))
        cols = {k: v[keep] for k, v in cols.items()}
        labels = labels[keep]
        clusters = clusters[keep]
    rng = np.random.default_rng(0)
    methods = sorted(set(labels.tolist()))
    rows = []
    for s in surro:
        for m in mets:
            rho, lo, hi = _rho_ci(cols[s], cols[m], clusters, rng)
            rows.append({"surrogate": s, "metric": m, "scope": "pooled",
                         "n": len(labels), "rho": rho, "ci_lo": lo, "ci_hi": hi})
            within = []
            for meth in methods:
                sel = labels == meth
                if sel.sum() >= 8:
                    within.append(spearmanr(cols[s][sel], cols[m][sel]).statistic)
            rows.append({"surrogate": s, "metric": m, "scope": "within-method-mean",
                         "n": len(labels),
                         "rho": float(np.nanmean(within)) if within else float("nan"),
                         "ci_lo": float(np.nanmin(within)) if within else float("nan"),
                         "ci_hi": float(np.nanmax(within)) if within else float("nan")})

    out_csv = os.path.join(args.results_dir, f"surrogate_alignment{args.suffix}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}")
    for r in rows:
        print(f"  {r['surrogate']:>14} vs {r['metric']:<11} [{r['scope']:>18}] "
              f"rho={r['rho']:+.3f}  ({r['ci_lo']:+.3f},{r['ci_hi']:+.3f})  n={r['n']}")

    # Scatter grid (surrogates x metrics), colored by method.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIX Two Text",
                       "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(len(surro), len(mets),
                             figsize=(5.5, 3.35),
                             squeeze=False)
    for i, s in enumerate(surro):
        for j, m in enumerate(mets):
            ax = axes[i][j]
            for meth in methods:
                sel = labels == meth
                ax.scatter(cols[s][sel], cols[m][sel], s=6, alpha=0.55,
                           color=PAPER_COLORS.get(meth, "0.4"),
                           label=PAPER_METHODS.get(meth, meth)
                           if (i == 0 and j == 0) else None)
            ax.set_xlabel(_display_surrogate(s))
            ax.set_ylabel(_display_metric(m, mets))
            ax.grid(color="0.92", linewidth=0.5, zorder=0)
    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=5,
               frameon=False, markerscale=1.8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=0.8, h_pad=0.8)
    os.makedirs(os.path.join(args.results_dir, "figures"), exist_ok=True)
    out_pdf = os.path.join(args.results_dir, "figures", f"surrogate_alignment{args.suffix}.pdf")
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
