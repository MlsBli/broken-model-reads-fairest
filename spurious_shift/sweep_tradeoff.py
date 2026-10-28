"""
sweep_tradeoff.py — Accuracy–fairness tradeoff curves by sweeping fairness strength.

One operating point per (method, scale): Fair-DER++ and FAIR-GEM scale their
margin parameters, FSW scales its gradient-adjustment alpha; every other
hyperparameter stays at its tuned value. Full 6-task training per point,
final-model metrics averaged over all tasks. Writes runs/seed<N>/tradeoff_curves.csv.

Usage (from spurious_shift/results): python ../sweep_tradeoff.py --seed N
After all seeds: python ../aggregate.py --tradeoff --seeds 0 1 2 3 4 5 6 7 8 9
"""

import csv
import os
import sys

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

import runctx
import config as cfg
from datasets.spurious_shift import get_spurious_tasks
from fairderpp import FairDERpp
from fairgem import FAIRGEM
from fsw import FSW
from metrics import accuracy_parity_gap, equalized_odds_gap, per_group_accuracy

SEED_ARG       = runctx.parse_seed(sys.argv)
EFFECTIVE_SEED = runctx.apply_seed(SEED_ARG)
DATA_SEED      = runctx.data_seed(SEED_ARG)
OUT            = runctx.output_dir(SEED_ARG)

SCALES_PROJ = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
SCALES_FSW  = [0.25, 0.5, 1.0, 2.0, 4.0]


def build(method, k, input_dim):
    if method == "fair_derpp":
        return FairDERpp(input_dim, device=cfg.DEVICE,
                         eps_alpha=k * cfg.FAIRDERPP_EPS_ALPHA,
                         eps_beta=k * cfg.FAIRDERPP_EPS_BETA)
    if method == "fair_gem":
        return FAIRGEM(input_dim, device=cfg.DEVICE, use_topk_dro=False,
                       use_disparity=True, dynamic_eps=True,
                       alpha=k * cfg.FAIRGEM_ALPHA, beta=k * cfg.FAIRGEM_BETA)
    if method == "fsw":
        return FSW(input_dim, device=cfg.DEVICE, alpha=k * cfg.FSW_ALPHA)
    raise ValueError(method)


def main():
    tasks, _ = get_spurious_tasks(seed=DATA_SEED)
    input_dim = tasks[0]["X_train"].shape[1]
    rows = []
    plan = [("fair_derpp", k) for k in SCALES_PROJ] + \
           [("fair_gem", k) for k in SCALES_PROJ] + \
           [("fsw", k) for k in SCALES_FSW]
    for method, k in plan:
        torch.manual_seed(EFFECTIVE_SEED)
        np.random.seed(EFFECTIVE_SEED)
        m = build(method, k, input_dim)
        for task in tasks:
            m.train_task(task["X_train"], task["y_train"], task["s_train"])
            m.remember(task["X_train"], task["y_train"], task["s_train"])
        bals, aps, eos, wgs = [], [], [], []
        for task in tasks:
            yv, sv = task["y_test"], task["s_test"]
            p = m.predict(task["X_test"])
            bals.append(balanced_accuracy_score(yv, p))
            aps.append(accuracy_parity_gap(yv, p, sv))
            eos.append(equalized_odds_gap(yv, p, sv))
            pg = per_group_accuracy(yv, p, sv, min_group_size=100)
            wgs.append(min(pg.values()) if pg else np.nan)
        rows.append(dict(method=method, scale=k,
                         bal=float(np.nanmean(bals)), acc_parity=float(np.nanmean(aps)),
                         eo=float(np.nanmean(eos)), worst=float(np.nanmean(wgs))))
        print(f"{method:>11} k={k:<5} bal={rows[-1]['bal']:.3f} "
              f"AP={rows[-1]['acc_parity']:.3f} EO={rows[-1]['eo']:.3f}")
    os.makedirs(OUT or ".", exist_ok=True)
    out = os.path.join(OUT or ".", "tradeoff_curves.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
