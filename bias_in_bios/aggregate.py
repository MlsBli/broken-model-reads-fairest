"""
aggregate.py — combine per-seed Bias-in-Bios runs into cross-seed aggregates.

Reads runs/seed*/<file>.npz produced by `train.py --seed N` (and by the
post-training eval scripts run under the same seed) and writes, per protocol:

    <stem>_agg.npz       per-metric mean and std across seeds, same matrix
                         shapes as a single run, plus the seed list
    <stem>_summary.csv   flat one-row-per-method table of the headline numbers

The per-seed arrays themselves stay under runs/seed<N>/.

Backward transfer needs the diagonal of the lower-triangular matrix, so it is
computed per seed and then averaged — not computed from the averaged matrix.
Averaging first would understate the spread, since BWT is a difference of two
correlated entries within the same run.

`eo` is the equalized-odds gap; the accuracy-parity gap is stored separately
as `acc_parity`.

Usage (run from bias_in_bios/results):
    python ../aggregate.py                     # results.npz (the CL trajectory)
    python ../aggregate.py --all               # every protocol found
    python ../aggregate.py --file results_retune.npz
    python ../aggregate.py --all --seeds 0 1 2 # use exactly these seed directories
"""

from __future__ import annotations

import csv
import glob
import os
import sys
import warnings

import numpy as np

RUNS_GLOB = os.path.join("runs", "seed*")

# Protocols worth aggregating, in reporting order. The deployed head is primary;
# the probes are diagnostics (see README).
PROTOCOLS = [
    "results.npz",
    "results_test.npz",
    "results_retune.npz",
    "results_retune_fair.npz",
    "results_probe_shared.npz",
    "results_probe_shared_fair.npz",
    "results_knn_shared.npz",
    "results_ncm_full.npz",
]

# (npz key, CSV label, higher_is_better)
METRIC_KEYS = [
    ("acc",         "Acc",         True),
    ("bal_acc",     "BalAcc",      True),
    ("worst_group", "WorstGroup",  True),
    ("eo",          "EO",          False),
    ("acc_parity",  "AccParity",   False),
    ("dp",          "DP",          False),
    ("tpr",         "TPRGap",      False),
    ("loss_gap",      "LossGap",     False),
    ("loss_gap_cond", "LossGapCond", False),
]


def _requested_seeds(argv: list[str]) -> list[int] | None:
    if "--seeds" not in argv:
        return None
    start = argv.index("--seeds") + 1
    values = []
    for arg in argv[start:]:
        if arg.startswith("--"):
            break
        values.append(int(arg))
    if not values:
        raise SystemExit("--seeds requires at least one integer")
    return values


def _seed_dirs(requested: list[int] | None = None) -> list[str]:
    if requested is not None:
        return [os.path.join("runs", f"seed{seed}") for seed in requested]
    dirs = sorted(
        glob.glob(RUNS_GLOB),
        key=lambda p: int(os.path.basename(p).removeprefix("seed")),
    )
    return [d for d in dirs if os.path.isdir(d)]


def _sample_std(vals) -> float:
    """Sample std (ddof=1); NaN for a single seed rather than a fake 0.0."""
    vals = np.asarray(vals, dtype=float)
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")


def _backward_transfer(matrix: np.ndarray, *, fairness: bool = False) -> float:
    """Mean final-vs-learned change; reverse the sign for fairness gaps."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[0] < 2:
        return float("nan")
    final = matrix[-1, :-1]
    learned = np.diag(matrix)[:-1]
    valid = np.isfinite(final) & np.isfinite(learned)
    if not valid.any():
        return float("nan")
    delta = learned[valid] - final[valid] if fairness else final[valid] - learned[valid]
    return float(delta.mean())


def _final_row_mean(matrix: np.ndarray) -> float:
    """Mean over tasks evaluated after the last task — the headline number."""
    T = matrix.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(matrix[T - 1, :T]))


def _write_runtime_summary(loaded, methods, seeds) -> None:
    if not all("runtime" in data.files for _, data in loaded):
        print("  warning: runtime data missing; runtime_summary.csv not written")
        return
    totals = {
        method: np.array([
            np.nansum(data["runtime"].item()[method]) for _, data in loaded
        ])
        for method in methods
    }
    references = {"fair_gem": "agem", "fair_derpp": "derpp"}
    fields = [
        "method", "n_seeds", "seeds", "total_seconds_mean", "total_seconds_std",
        "reference", "overhead_ratio_mean", "overhead_ratio_std",
    ]
    with open("runtime_summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for method in methods:
            reference = references.get(method, "")
            ratios = (totals[method] / totals[reference]
                      if reference and np.all(totals[reference] > 0) else np.array([]))
            writer.writerow({
                "method": method,
                "n_seeds": len(seeds),
                "seeds": " ".join(map(str, seeds)),
                "total_seconds_mean": f"{np.mean(totals[method]):.3f}",
                "total_seconds_std": f"{_sample_std(totals[method]):.3f}",
                "reference": reference,
                "overhead_ratio_mean": f"{np.mean(ratios):.4f}" if ratios.size else "",
                "overhead_ratio_std": f"{_sample_std(ratios):.4f}" if ratios.size else "",
            })
    print("Wrote runtime_summary.csv")


def aggregate(filename: str, requested: list[int] | None = None) -> int:
    stem = filename.removesuffix(".npz")
    out_npz, out_csv = f"{stem}_agg.npz", f"{stem}_summary.csv"

    seed_dirs = _seed_dirs(requested)
    if not seed_dirs:
        print(f"No per-seed runs under {RUNS_GLOB}/. Run: python ../train.py --seed N")
        return 1

    absent_dirs = [d for d in seed_dirs if not os.path.isdir(d)]
    if absent_dirs:
        print(f"ERROR: requested seed directories absent: {', '.join(absent_dirs)}")
        return 1

    loaded = []
    for d in seed_dirs:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            loaded.append((d, np.load(path, allow_pickle=True)))
    if not loaded:
        print(f"  {filename}: not found in any seed directory - skipping")
        return 0

    methods = [str(m) for m in loaded[0][1]["methods"]]
    states = [str(s) for s in loaded[0][1]["states"]]
    seeds = [int(os.path.basename(d).removeprefix("seed")) for d, _ in loaded]
    print(f"\n{filename}: {len(loaded)} seeds {seeds}")

    agg: dict[str, np.ndarray] = {}
    mean_metrics: dict[str, dict[str, np.ndarray]] = {}
    summary: dict[str, dict[str, tuple[float, float]]] = {m: {} for m in methods}
    present = []

    for key, label, _hib in METRIC_KEYS:
        if key not in loaded[0][1].files:
            continue
        present.append(label)
        mean_metrics[key] = {}
        for m in methods:
            stack = np.stack([data[key].item()[m] for _, data in loaded])
            # The upper triangle is NaN in every seed by construction — a task is
            # never evaluated before it is trained on. nanmean/nanstd correctly
            # return NaN there; silence the warning so real problems stay visible.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                agg[f"{key}__{m}__mean"] = np.nanmean(stack, axis=0)
                agg[f"{key}__{m}__std"] = np.nanstd(stack, axis=0)
            mean_metrics[key][m] = agg[f"{key}__{m}__mean"]
            # Seed-level values, kept so paired tests over seeds (e.g. an exact
            # permutation test) are possible downstream.
            agg[f"{key}__{m}__per_seed"] = stack
            per_seed = [_final_row_mean(s) for s in stack]
            summary[m][label] = (float(np.mean(per_seed)), _sample_std(per_seed))

    # Transfer metrics: computed within each seed, then averaged.
    has_transfer = all(k in loaded[0][1].files for k in ("bal_acc", "eo"))
    if has_transfer:
        for m in methods:
            bwt = [_backward_transfer(d["bal_acc"].item()[m]) for _, d in loaded]
            fbwt = [
                _backward_transfer(d["eo"].item()[m], fairness=True)
                for _, d in loaded
            ]
            if not np.all(np.isnan(bwt)):
                summary[m]["Acc-BWT"] = (float(np.nanmean(bwt)), float(np.nanstd(bwt)))
                summary[m]["F-BWT"] = (float(np.nanmean(fbwt)), float(np.nanstd(fbwt)))
        if "Acc-BWT" in summary[methods[0]]:
            present += ["Acc-BWT", "F-BWT"]

    np.savez(out_npz, seeds=np.array(seeds), methods=methods, states=states,
             **mean_metrics, **agg)

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["method", "n_seeds"]
                        + [c for col in present for c in (f"{col}_mean", f"{col}_std")])
        for m in methods:
            row = [m, len(loaded)]
            for col in present:
                mean, std = summary[m][col]
                row += [f"{mean:.4f}", f"{std:.4f}"]
            writer.writerow(row)

    if filename == "results.npz":
        _write_runtime_summary(loaded, methods, seeds)

    print(f"{'Method':<16}" + "".join(f"{c:>18}" for c in present))
    print("-" * (16 + 18 * len(present)))
    for m in methods:
        print(f"{m:<16}" + "".join(
            f"{summary[m][c][0]:>11.3f} +/-{summary[m][c][1]:<5.3f}" for c in present))
    print(f"Wrote {out_npz} and {out_csv}")
    return 0


def main() -> int:
    requested = _requested_seeds(sys.argv)
    if "--all" in sys.argv:
        rc = 0
        for f in PROTOCOLS:
            rc |= aggregate(f, requested)
        return rc
    for i, arg in enumerate(sys.argv):
        if arg == "--file" and i + 1 < len(sys.argv):
            return aggregate(sys.argv[i + 1], requested)
    return aggregate("results.npz", requested)


if __name__ == "__main__":
    sys.exit(main())
