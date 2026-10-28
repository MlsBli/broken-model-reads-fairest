"""fsw_reference.py — FSW at the reference paper's fixed hyperparameters.

Our FSW numbers come from a tuned configuration selected under the same protocol
as every other method. This study asks the separate question of how FSW behaves
at the hyperparameters its own paper reports, which is a fairness-to-the-baseline
control rather than a competitive comparison.

By default it runs over the seeds of the main results (n_seeds in
spurious_shift/results/results_summary.csv), with the same task construction,
lower-triangular evaluation and final-row summary convention, so the row is
directly comparable to the main table.

Usage:
    python analysis/fsw_reference.py                 # seeds of the main results
    python analysis/fsw_reference.py --seeds 0 1 2   # explicit seed list
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parent.parent
SPUR = ROOT / "spurious_shift"
RESULTS = SPUR / "results"
OUT = RESULTS / "fsw_reference_summary.csv"
sys.path.insert(0, str(SPUR))

# The reference paper's reported setting, not ours.
REF_ALPHA, REF_LAMBDA, REF_TAU = 0.002, 0.5, 2.0


def campaign_seeds() -> list[int]:
    with (RESULTS / "results_summary.csv").open(newline="", encoding="utf-8") as fh:
        return list(range(int(next(csv.DictReader(fh))["n_seeds"])))


def final_row_mean(matrix: np.ndarray) -> float:
    m = np.asarray(matrix, dtype=float)
    return float(np.nanmean(m[m.shape[0] - 1, : m.shape[0]]))


def run_seed(seed: int) -> dict[str, float]:
    import torch
    from sklearn.metrics import balanced_accuracy_score
    import runctx
    from datasets.spurious_shift import get_spurious_tasks
    from metrics import (accuracy_parity_gap, equalized_odds_gap,
                         per_group_accuracy)
    from fsw import FSW
    from config import DEVICE, SPURIOUS_N_TASKS

    runctx.apply_seed(seed)
    tasks, _ = get_spurious_tasks(seed=runctx.data_seed(seed))
    T = SPURIOUS_N_TASKS
    model = FSW(tasks[0]["X_train"].shape[1], device=DEVICE,
                alpha=REF_ALPHA, lmbd=REF_LAMBDA, tau=REF_TAU)

    bal = np.full((T, T), np.nan)
    ap = np.full((T, T), np.nan)
    eo = np.full((T, T), np.nan)
    worst = np.full((T, T), np.nan)
    for t in range(T):
        task = tasks[t]
        model.train_task(task["X_train"], task["y_train"], task["s_train"])
        model.remember(task["X_train"], task["y_train"], task["s_train"])
        for e in range(t + 1):
            et = tasks[e]
            preds = model.predict(et["X_test"])
            y, s = et["y_test"], et["s_test"]
            bal[t, e] = balanced_accuracy_score(y, preds)
            ap[t, e] = accuracy_parity_gap(y, preds, s)
            eo[t, e] = equalized_odds_gap(y, preds, s)
            pg = per_group_accuracy(y, preds, s)
            vals = [v for v in pg.values() if v == v]
            worst[t, e] = min(vals) if vals else np.nan
    return {"BalAcc": final_row_mean(bal), "AccParity": final_row_mean(ap),
            "EO": final_row_mean(eo), "WorstGroup": final_row_mean(worst)}


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--seeds", nargs="*", type=int)
    args = ap_.parse_args()
    seeds = args.seeds if args.seeds else campaign_seeds()

    per_seed: dict[str, list[float]] = {}
    for s in seeds:
        got = run_seed(s)
        print(f"  seed {s}: " + "  ".join(f"{k}={v:.4f}" for k, v in got.items()))
        for k, v in got.items():
            per_seed.setdefault(k, []).append(v)

    row = {"protocol": "fixed_reference", "alpha": REF_ALPHA,
           "lambda": REF_LAMBDA, "tau": REF_TAU, "n_seeds": len(seeds)}
    for k, vals in per_seed.items():
        row[f"{k}_mean"] = f"{np.mean(vals):.4f}"
        row[f"{k}_std"] = f"{np.std(vals, ddof=1):.4f}"
    fields = ["protocol", "alpha", "lambda", "tau", "n_seeds"]
    for k in ("BalAcc", "AccParity", "EO", "WorstGroup"):
        fields += [f"{k}_mean", f"{k}_std"]
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerow(row)
    print(f"Wrote {OUT.relative_to(ROOT)} over {len(seeds)} seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
