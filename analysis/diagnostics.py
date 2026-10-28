#!/usr/bin/env python3
"""
diagnostics.py - per-step mechanism and representation-probe tables.

Reads the per-seed arrays under <suite>/results/runs/ and writes, under
analysis/results/:
    mechanism_diagnostics.csv  safety-gradient statistics per method and task
    probe_diagnostics.csv      Bias in Bios nearest-class-mean and leakage probes

    python analysis/diagnostics.py [--benchmark bios] [--seeds 0 1 2 ...]
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


OURS = {"fair_gem", "fair_derpp"}


METRICS = {
    "spurious": {
        "BalAcc": ("bal_acc", "up"),
        "Acc-BWT": ("acc_bwt", "up"),
        "AccParity": ("eo", "down"),
        "TopRate": ("dp", "down"),
        "F-BWT": ("fair_bwt", "up"),
        "EO": ("eo_true", "down"),
        "WorstGroup": ("worst_group", "up"),
    },
    "bios": {
        "BalAcc": ("bal_acc", "up"),
        "WorstGroup": ("worst_group", "up"),
        "TPRGap": ("tpr", "down"),
        "EO": ("eo", "down"),
        "DP": ("dp", "down"),
        "AccParity": ("acc_parity", "down"),
    },
}


def _code_dir(root: Path, benchmark: str) -> Path:
    if benchmark == "spurious":
        return root / "spurious_shift" / "results"
    if benchmark == "bios":
        return root / "bias_in_bios" / "results"
    raise ValueError(benchmark)


def _fmt(x) -> str:
    if x is None:
        return ""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(v):
        return ""
    return f"{v:.6g}"


def _final_mean(matrix: np.ndarray) -> float:
    t = matrix.shape[0] - 1
    return float(np.nanmean(matrix[t, : t + 1]))


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _alignment_values(result_path: Path, benchmark: str,
                      task_idx: int) -> dict[str, str]:
    if not result_path.exists():
        return {}
    data = np.load(result_path, allow_pickle=True)
    out = {}
    t_final = len(data["states"]) - 1
    for label, (key, _direction) in METRICS[benchmark].items():
        if key in {"acc_bwt", "fair_bwt"} or key not in data.files:
            continue
        values = data[key].item()
        for method in sorted(OURS & values.keys()):
            mat = values[method]
            if task_idx < mat.shape[1]:
                out[f"final_{method}_{label}"] = _fmt(mat[t_final, task_idx])
    return out


def _run_sources(code: Path, expected_seeds: set[int] | None,
                 filename: str) -> list[tuple[int | None, Path]]:
    if expected_seeds is not None:
        return [
            (seed, code / "runs" / f"seed{seed}" / filename)
            for seed in sorted(expected_seeds)
        ]
    sources = []
    for path in sorted((code / "runs").glob(f"seed*/{filename}")):
        seed = int(path.parent.name.removeprefix("seed"))
        sources.append((seed, path))
    if not sources and (code / filename).exists():
        sources.append((None, code / filename))
    return sources


def write_mechanism_diagnostics(root: Path, benchmarks: set[str],
                                expected_seeds: set[int] | None = None) -> Path:
    out = root / "analysis" / "results" / "mechanism_diagnostics.csv"
    existing = _read_csv(out)
    rows = []
    for benchmark in ("spurious", "bios"):
        if benchmark not in benchmarks:
            continue
        code = _code_dir(root, benchmark)
        sources = [
            (seed, path)
            for seed, path in _run_sources(
                code, expected_seeds, "results_proj_stats.npy"
            )
            if path.exists()
        ]
        if not sources:
            # No per-seed diagnostic arrays for this benchmark: keep its
            # existing rows in the CSV rather than erasing them.
            rows.extend(row for row in existing if row.get("benchmark") == benchmark)
            continue
        # If the per-seed files lack keys that the existing CSV has (e.g. the
        # realized-step rates), rewriting would blank those columns; keep the
        # existing rows instead.
        expected_keys = {
            key for row in existing if row.get("benchmark") == benchmark
            for key, value in row.items() if value not in ("", None)
        }
        available_keys = set()
        for _seed, path in sources:
            stats_probe = np.load(path, allow_pickle=True).item()
            for by_task in stats_probe.values():
                for per_task in by_task.values():
                    available_keys |= set(per_task)
        lost = {
            key for key in expected_keys - available_keys
            if key not in {"benchmark", "seed", "method", "task"}
            and not key.startswith("final_")
        }
        if lost:
            print(f"  {benchmark}: per-seed diagnostics lack {sorted(lost)}; "
                  f"keeping the committed rows rather than blanking them")
            rows.extend(row for row in existing
                        if row.get("benchmark") == benchmark)
            continue

        for seed, path in sources:
            stats = np.load(path, allow_pickle=True).item()
            result_path = path.parent / "results_test.npz"
            if not result_path.exists():
                result_path = path.parent / "results.npz"
            for method, by_task in stats.items():
                for task_idx in sorted(by_task):
                    row = {
                        "benchmark": benchmark,
                        "seed": "" if seed is None else seed,
                        "method": method,
                        "task": task_idx,
                    }
                    for key, value in sorted(by_task[task_idx].items()):
                        row[key] = _fmt(value)
                    row.update(_alignment_values(result_path, benchmark, int(task_idx)))
                    rows.append(row)
    rows = _preserve_other_benchmarks(out, rows, benchmarks)
    _write_csv(out, rows, list(existing[0]) if existing else [
        "benchmark", "seed", "method", "task", "best_group_mode", "worst_group_mode",
        "best_group_switch_rate", "worst_group_switch_rate", "loss_gap_mean",
        "loss_gap_std", "loss_gap_p10", "loss_gap_p50", "loss_gap_p90",
        "safety_proj_rate", "mem_proj_rate", "mean_eps_t", "mean_dro_loss",
        "mean_dot_safety", "n_steps",
    ])
    return out


def write_probe_diagnostics(root: Path, benchmarks: set[str],
                            expected_seeds: set[int] | None = None) -> Path:
    """Flatten the Bias in Bios NCM and leakage studies into a per-seed table."""
    rows = []
    if "bios" in benchmarks:
        code = _code_dir(root, "bios")
        for seed, path in _run_sources(code, expected_seeds, "results_ncm_full.npz"):
            if not path.exists():
                continue
            ncm = np.load(path, allow_pickle=True)
            for label, (key, _direction) in METRICS["bios"].items():
                if key not in ncm.files:
                    continue
                for method, matrix in ncm[key].item().items():
                    rows.append({
                        "benchmark": "bios", "seed": "" if seed is None else seed,
                        "study": "ncm", "method": method, "metric": label,
                        "mean": _fmt(_final_mean(matrix)),
                    })

        for seed, path in _run_sources(code, expected_seeds, "results_leakage.npy"):
            if not path.exists():
                continue
            leakage = np.load(path, allow_pickle=True).item()
            groups = {"frozen_bert_input": leakage.get("baseline", {})}
            groups.update(leakage.get("methods", {}))
            for method, by_task in groups.items():
                values = list(by_task.values())
                for metric in ("acc", "auc"):
                    metric_values = [v[metric] for v in values if metric in v]
                    if metric_values:
                        rows.append({
                            "benchmark": "bios", "seed": "" if seed is None else seed,
                            "study": "leakage", "method": method, "metric": metric,
                            "mean": _fmt(np.mean(metric_values)),
                        })

    out = root / "analysis" / "results" / "probe_diagnostics.csv"
    rows = _preserve_other_benchmarks(out, rows, benchmarks)
    _write_csv(out, rows, [
        "benchmark", "seed", "study", "method", "metric", "mean",
    ])
    return out


_BENCH_ORDER = {"spurious": 0, "bios": 1}


def _preserve_other_benchmarks(path: Path, rows: list[dict],
                               benchmarks: set[str]) -> list[dict]:
    existing = [
        row for row in _read_csv(path)
        if row.get("benchmark") and row.get("benchmark") not in benchmarks
    ]
    # Sort by benchmark so the output does not depend on the order in which
    # benchmarks were processed.
    merged = existing + rows
    return sorted(merged, key=lambda r: _BENCH_ORDER.get(r.get("benchmark", ""), 9))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    keys = list(fieldnames or [])
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    # The output directory may not exist yet.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Write the mechanism and probe diagnostic tables.")
    parser.add_argument("--benchmark", choices=["all", "spurious", "bios"], default="all")
    parser.add_argument("--seeds", nargs="+", type=int,
                        help="Use exactly these seeds (default: every seed found).")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    benchmarks = {"spurious", "bios"} if args.benchmark == "all" else {args.benchmark}
    seeds = set(args.seeds) if args.seeds else None
    for path in (write_mechanism_diagnostics(root, benchmarks, seeds),
                 write_probe_diagnostics(root, benchmarks, seeds)):
        print(f"Wrote {path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
