"""
aggregate.py — combine per-seed runs into cross-seed aggregates.

Reads every runs/seed*/results*.npz produced by `train.py --seed N` and writes:

    results_agg.npz      per-metric mean and std across seeds, same matrix
                         shapes as a single run, plus the seed list
    results_summary.csv  flat one-row-per-method table of the headline numbers

Backward transfer needs the diagonal of the lower-triangular matrix, so it is
computed per seed and then averaged — not computed from the averaged matrix.
Averaging first would understate the spread, since BWT is a difference of two
correlated entries within the same run.

Usage (run from spurious_shift/results):
    python ../aggregate.py                  # aggregate results.npz across seeds
    python ../aggregate.py --test           # aggregate results_test.npz instead
    python ../aggregate.py --sync-test-view # derive the final-row test view
    python ../aggregate.py --ablations      # aggregate results_ablations.npz
    python ../aggregate.py --tradeoff       # aggregate fairness-strength sweeps
    python ../aggregate.py --seeds 0 1 2    # use exactly these seed directories
"""

from __future__ import annotations

import csv
import glob
import os
import sys
import warnings

import numpy as np

RUNS_GLOB = os.path.join("runs", "seed*")

# (key in npz, human label, direction) — direction is only used for the CSV header.
# 'eo' holds the accuracy-parity gap (what the paper reports); 'eo_true' is the
# textbook equalized-odds gap. 'worst_group' is carried so a small gap achieved
# by collapsing toward chance is visible rather than flattering.
METRIC_KEYS = [
    ("bal_acc",     "BalAcc",      "up"),
    ("eo",          "AccParity",   "down"),
    ("dp",          "TopRate",     "down"),
    ("eo_true",     "EO",          "down"),
    ("worst_group", "WorstGroup",  "up"),
    ("loss_gap",      "LossGap",     False),
    ("loss_gap_cond", "LossGapCond", False),
]

# Optional metrics: result files without them are aggregated without these
# columns.
OPTIONAL_KEYS = {"eo_true", "worst_group", "loss_gap", "loss_gap_cond"}

TRADEOFF_METRICS = ("bal", "acc_parity", "eo", "worst")


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


def _load(seed_dir: str, filename: str):
    path = os.path.join(seed_dir, filename)
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)


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


def aggregate_tradeoff(requested: list[int] | None = None) -> int:
    """Aggregate one fairness-strength row per method, scale, and seed."""
    seed_dirs = _seed_dirs(requested)
    if not seed_dirs:
        print(f"No per-seed runs found under {RUNS_GLOB}/.")
        return 1

    rows_by_arm: dict[tuple[str, float], list[dict[str, str]]] = {}
    expected_arms: set[tuple[str, float]] | None = None
    seeds = []
    for seed_dir in seed_dirs:
        path = os.path.join(seed_dir, "tradeoff_curves.csv")
        if not os.path.isfile(path):
            print(f"ERROR: missing {path}")
            return 1
        seed = int(os.path.basename(seed_dir).removeprefix("seed"))
        seeds.append(seed)
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        arms = {(row["method"], float(row["scale"])) for row in rows}
        if len(arms) != len(rows):
            print(f"ERROR: duplicate method/scale rows in {path}")
            return 1
        if expected_arms is None:
            expected_arms = arms
        elif arms != expected_arms:
            print(f"ERROR: inconsistent method/scale grid in {path}")
            return 1
        for row in rows:
            rows_by_arm.setdefault((row["method"], float(row["scale"])), []).append(row)

    fields = ["method", "scale", "n", "seeds"] + [
        field for metric in TRADEOFF_METRICS
        for field in (f"{metric}_mean", f"{metric}_std")
    ]
    with open("tradeoff_curves_agg.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for method, scale in sorted(rows_by_arm, key=lambda arm: (arm[0], arm[1])):
            arm_rows = rows_by_arm[(method, scale)]
            out = {
                "method": method, "scale": scale, "n": len(arm_rows),
                "seeds": " ".join(map(str, seeds)),
            }
            for metric in TRADEOFF_METRICS:
                values = [float(row[metric]) for row in arm_rows]
                out[f"{metric}_mean"] = f"{np.mean(values):.4f}"
                out[f"{metric}_std"] = f"{_sample_std(values):.4f}"
            writer.writerow(out)
    print(f"Wrote tradeoff_curves_agg.csv from seeds {seeds}")
    return 0


def _final_row_mean(matrix: np.ndarray) -> float:
    """Mean over the tasks evaluated after the last task — the headline number."""
    T = matrix.shape[0]
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
    references = {"fair_gem": "agem", "fair_derpp": "der_pp"}
    fields = [
        "method", "n_seeds", "seeds", "total_seconds_mean", "total_seconds_std",
        "reference", "overhead_ratio_mean", "overhead_ratio_std",
    ]
    with open("runtime_summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
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
                "total_seconds_std": f"{np.std(totals[method]):.3f}",
                "reference": reference,
                "overhead_ratio_mean": f"{np.mean(ratios):.4f}" if ratios.size else "",
                "overhead_ratio_std": f"{np.std(ratios):.4f}" if ratios.size else "",
            })
    print("Wrote runtime_summary.csv")


def aggregate(filename: str, out_npz: str, out_csv: str,
              requested: list[int] | None = None) -> int:
    seed_dirs = _seed_dirs(requested)
    if not seed_dirs:
        print(f"No per-seed runs found under {RUNS_GLOB}/. "
              f"Run: python ../train.py --seed N")
        return 1

    absent_dirs = [d for d in seed_dirs if not os.path.isdir(d)]
    if absent_dirs:
        print(f"ERROR: requested seed directories absent: {', '.join(absent_dirs)}")
        return 1

    loaded = [(d, _load(d, filename)) for d in seed_dirs]
    missing = [d for d, data in loaded if data is None]
    loaded = [(d, data) for d, data in loaded if data is not None]
    if missing:
        print(f"  warning: {filename} absent in {', '.join(missing)} - skipping those")
    if not loaded:
        print(f"No {filename} found in any seed directory.")
        return 1

    methods = [str(m) for m in loaded[0][1]["methods"]]
    states = [str(s) for s in loaded[0][1]["states"]]
    seeds = []
    for d, data in loaded:
        seeds.append(int(data["seed"]) if "seed" in data.files
                     else int(os.path.basename(d).removeprefix("seed")))

    print(f"Aggregating {filename} over {len(loaded)} seeds: {seeds}")

    agg: dict[str, np.ndarray] = {}
    mean_metrics: dict[str, dict[str, np.ndarray]] = {}
    summary: dict[str, dict[str, tuple[float, float]]] = {m: {} for m in methods}

    present_labels = []
    for key, label, _dir in METRIC_KEYS:
        if key not in loaded[0][1].files:
            if key in OPTIONAL_KEYS:
                continue  # optional metric not present in this file
            print(f"  warning: {key!r} missing from {filename} - skipping")
            continue
        present_labels.append(label)
        mean_metrics[key] = {}
        # Stack matrices: (n_seeds, T, T) per method.
        for m in methods:
            stack = np.stack([data[key].item()[m] for _, data in loaded])
            # The upper triangle is NaN in every seed by construction — a task is
            # never evaluated before it is trained on. nanmean/nanstd correctly
            # return NaN there; silence the warning so real problems stay visible.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                agg[f"{key}__{m}__mean"] = np.nanmean(stack, axis=0)
                # ddof=1 to match the summary CSV's _sample_std; NaN with a
                # single seed, which has no measured spread.
                agg[f"{key}__{m}__std"] = np.nanstd(stack, axis=0, ddof=1)
            # The per-seed stack itself. A statistic of several cells (a row
            # mean, BWT, an accuracy-weighted gap) cannot take its error bar
            # from the per-cell stds above, since cells within a seed are
            # correlated; evaluate it within each seed and take the spread
            # across seeds instead.
            agg[f"{key}__{m}__per_seed"] = stack
            mean_metrics[key][m] = agg[f"{key}__{m}__mean"]
            per_seed = [_final_row_mean(s) for s in stack]
            summary[m][label] = (float(np.mean(per_seed)), _sample_std(per_seed))

    # Transfer metrics: computed within each seed, then averaged.
    #
    # Both need the diagonal, which results_test.npz does not carry (only its
    # final row is populated), so they are undefined there. The columns are
    # dropped when no seed could compute them.
    transfer_defined = False
    for m in methods:
        bwt_per_seed, fbwt_per_seed = [], []
        for _, data in loaded:
            bwt_per_seed.append(_backward_transfer(data["bal_acc"].item()[m]))
            fbwt_per_seed.append(_backward_transfer(
                data["eo"].item()[m], fairness=True,
            ))
        if np.any(np.isfinite(bwt_per_seed)) or np.any(np.isfinite(fbwt_per_seed)):
            transfer_defined = True
        summary[m]["Acc-BWT"] = (float(np.mean(bwt_per_seed)), _sample_std(bwt_per_seed))
        summary[m]["F-BWT"] = (float(np.mean(fbwt_per_seed)), _sample_std(fbwt_per_seed))
    if not transfer_defined:
        print("  note: no diagonal in this result file - Acc-BWT and F-BWT are "
              "undefined and are omitted rather than reported as zero")

    np.savez(out_npz, seeds=np.array(seeds), methods=methods, states=states,
             **mean_metrics, **agg)

    # Fixed leading columns, then whichever optional metrics this run carried.
    columns = ["BalAcc", "Acc-BWT", "AccParity", "TopRate", "F-BWT"]
    if not transfer_defined:
        columns = [c for c in columns if c not in ("Acc-BWT", "F-BWT")]
    columns += [l for l in present_labels if l in {"EO", "WorstGroup"}]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["method", "n_seeds"]
                        + [c for col in columns for c in (f"{col}_mean", f"{col}_std")])
        for m in methods:
            row = [m, len(loaded)]
            for col in columns:
                mean, std = summary[m][col]
                row += [f"{mean:.4f}", f"{std:.4f}"]
            writer.writerow(row)

    if filename == "results.npz":
        _write_runtime_summary(loaded, methods, seeds)

    print(f"\n{'Method':<16}" + "".join(f"{c:>18}" for c in columns))
    print("-" * (16 + 18 * len(columns)))
    for m in methods:
        cells = "".join(f"{summary[m][c][0]:>11.3f} ±{summary[m][c][1]:<5.3f}"
                        for c in columns)
        print(f"{m:<16}{cells}")

    print(f"\nWrote {out_npz} and {out_csv}")
    return 0


def sync_test_view() -> int:
    """Derive the final-row results_test_* view from results_agg."""
    source_path = "results_agg.npz"
    if not os.path.exists(source_path):
        print("No results_agg.npz found. Run aggregate.py first.")
        return 1
    source = np.load(source_path, allow_pickle=True)
    methods = [str(method) for method in source["methods"]]
    out = {
        "seeds": source["seeds"],
        "methods": source["methods"],
        "states": source["states"],
    }
    summary = {method: {} for method in methods}
    labels = []
    for key, label, _direction in METRIC_KEYS:
        if key not in source.files:
            continue
        labels.append(label)
        means = {}
        for method in methods:
            mean = np.asarray(source[f"{key}__{method}__mean"], dtype=float)
            std = np.asarray(source[f"{key}__{method}__std"], dtype=float)
            stack = np.asarray(source[f"{key}__{method}__per_seed"], dtype=float)
            mean_view = np.full_like(mean, np.nan)
            std_view = np.full_like(std, np.nan)
            stack_view = np.full_like(stack, np.nan)
            mean_view[-1, :] = mean[-1, :]
            std_view[-1, :] = std[-1, :]
            stack_view[:, -1, :] = stack[:, -1, :]
            means[method] = mean_view
            out[f"{key}__{method}__mean"] = mean_view
            out[f"{key}__{method}__std"] = std_view
            out[f"{key}__{method}__per_seed"] = stack_view
            per_seed = [
                float(np.nanmean(seed_matrix[-1, :])) for seed_matrix in stack
            ]
            summary[method][label] = (
                float(np.mean(per_seed)), _sample_std(per_seed)
            )
        out[key] = means

    np.savez("results_test_agg.npz", **out)
    columns = ["BalAcc", "AccParity", "TopRate"]
    columns += [label for label in labels if label in {"EO", "WorstGroup"}]
    with open("results_test_summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(
            ["method", "n_seeds"]
            + [field for label in columns for field in (f"{label}_mean", f"{label}_std")]
        )
        for method in methods:
            row = [method, len(source["seeds"])]
            for label in columns:
                mean, std = summary[method][label]
                row.extend((f"{mean:.4f}", f"{std:.4f}"))
            writer.writerow(row)
    print("Wrote results_test_agg.npz and results_test_summary.csv from results_agg.npz")
    return 0


def main() -> int:
    requested = _requested_seeds(sys.argv)
    if "--sync-test-view" in sys.argv:
        return sync_test_view()
    if "--tradeoff" in sys.argv:
        return aggregate_tradeoff(requested)
    if "--test" in sys.argv:
        return aggregate("results_test.npz", "results_test_agg.npz",
                         "results_test_summary.csv", requested)
    if "--ablations" in sys.argv:
        return aggregate("results_ablations.npz", "results_ablations_agg.npz",
                         "results_ablations_summary.csv", requested)
    return aggregate("results.npz", "results_agg.npz", "results_summary.csv",
                     requested)


if __name__ == "__main__":
    sys.exit(main())
