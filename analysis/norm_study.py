#!/usr/bin/env python3
"""
norm_study.py — is the normalisation choice a real architectural decision, or
something hyperparameter tuning absorbs?

Motivation. The disparity safety gradient is computed from a forward pass whose
normalisation we control. Under BatchNorm in train mode, activations are
normalised by statistics pooled over the batch, so gradients couple samples
across demographic groups and the safety gradient can act partly through that
coupling. Switching that pass to eval mode removes the channel. Comparing the
two at a *single* hyperparameter setting is not a fair test, because
`eps_t = min(alpha, beta * L_disp)` calibrates the fairness margin against the
scale of the disparity loss — and that scale is exactly what the normalisation
choice changes. Each condition must therefore be tuned on its own terms.

Conditions (each tuned independently, then evaluated at its own optimum):

  bn-legacy   BatchNorm, auxiliary passes update running statistics
  bn-freeze   BatchNorm, batch-statistics normalisation, running stats frozen
  bn-eval     BatchNorm, auxiliary passes use frozen running statistics
  layernorm   LayerNorm — no batch coupling and no running state at all

The last matters most for scope: Bias-in-Bios uses BERT, which is LayerNorm
throughout. If the mechanism only worked under "batch", it would be an
architectural dependency.

Usage:
    python analysis/norm_study.py --code-dir spurious_shift \
        --python <interp> --trials 20 --seeds 42
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# (id, CLFAIR_NORM, CLFAIR_BN_MODE)
CONDITIONS = [
    ("bn-legacy", "batch", "legacy"),
    ("bn-freeze", "batch", "freeze"),
    ("bn-eval",   "batch", "eval"),
    ("layernorm", "layer", "freeze"),   # mode is inert without running stats
]

# Every method with tunable parameters is retuned in every condition, not only
# the safety-gradient methods: FaIRL's replay pass also goes through
# batchnorm_eval, so it is affected by the BatchNorm mode. GroupDRO has no
# auxiliary pass, but the LayerNorm condition changes the architecture under
# every method, so it is retuned too.
#
# A-GEM is absent because, with the memory budget fixed, it has no free
# hyperparameters at all — the learning rate is shared via MLP_LR.
TUNE_METHODS = "GroupDRO,DER++,FAIRL,FAIR-GEM,Fair-DER++"

TRACKED = ["FAIRGEM_ALPHA", "FAIRGEM_BETA", "FAIRDERPP_ALPHA", "FAIRDERPP_BETA",
           "FAIRDERPP_EPS_ALPHA", "FAIRDERPP_EPS_BETA", "DERPP_ALPHA", "DERPP_BETA",
           "FAIRL_LAMBDA", "FAIRL_EPS", "DRO_ETA", "DRO_ADJ"]
METRICS = ["BalAcc", "Acc-BWT", "AccParity", "F-BWT"]
REPORT = ["agem", "der_pp", "group_dro", "fairl", "fair_gem", "fair_derpp"]


def _read_params(cfg: Path) -> dict[str, float]:
    t = cfg.read_text(encoding="utf-8")
    out = {}
    for k in TRACKED:
        m = re.search(rf"^{k}\s*=\s*([\d.eE+-]+)", t, re.M)
        if m:
            out[k] = float(m.group(1))
    return out


def _summarise(results: Path, code: Path, py: list[str]) -> dict[str, list[float]]:
    """
    Read one results.npz into {method: [BalAcc, Acc-BWT, AccParity, F-BWT]}.

    Runs in a subprocess on purpose. Every condition has its own copy of the
    code, all exporting a module named `metrics`; importing two of them into
    this process would resolve to whichever landed in sys.modules first, so
    conditions would silently score each other's results.
    """
    script = """
import json, sys
import numpy as np
from metrics import backward_transfer, fairness_backward_transfer
z = np.load(sys.argv[1], allow_pickle=True)
out = {}
for m in [str(x) for x in z["methods"]]:
    ba, eo = z["bal_acc"].item()[m], z["eo"].item()[m]
    T = ba.shape[0]
    out[m] = [float(np.nanmean(ba[T-1, :T])), backward_transfer(ba),
              float(np.nanmean(eo[T-1, :T])), fairness_backward_transfer(eo)]
print(json.dumps(out))
"""
    r = subprocess.run(py + ["-c", script, str(results)],
                       cwd=code, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"summarise failed:\n{r.stderr[-2000:]}")
    return json.loads(r.stdout)


def _run_condition(cid, norm, bn_mode, code, tmp, py, args, threads):
    work = Path(tmp) / cid
    shutil.copytree(code, work, ignore=shutil.ignore_patterns(
        "results", "runs", "figures", "__pycache__", "*.pt", "*.db", "*.pdf",
        "*.npz", "*.npy", "*.csv"))
    tune = work / "tune.py"
    tune.write_text(re.sub(r"^N_TRIALS\s*=\s*\d+", f"N_TRIALS = {args.trials}",
                           tune.read_text(encoding="utf-8"), flags=re.M), encoding="utf-8")

    # Cap per-condition threading. Torch otherwise grabs every core in each
    # process, so N conditions oversubscribe by N and all of them slow down.
    env = {**os.environ, "MPLBACKEND": "Agg",
           "CLFAIR_NORM": norm, "CLFAIR_BN_MODE": bn_mode,
           "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
           "TORCH_NUM_THREADS": str(threads)}

    if not args.skip_tuning:
        print(f"[{cid}] tuning {TUNE_METHODS} ({args.trials} trials, "
              f"norm={norm} bn={bn_mode})", flush=True)
        r = subprocess.run(py + ["tune.py", "--fresh", "--methods", TUNE_METHODS],
                           cwd=work, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            return cid, None, None, f"tuning failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}"

    params = _read_params(work / "config.py")
    print(f"[{cid}] tuned: " + json.dumps(params, sort_keys=True), flush=True)

    per_seed = []
    for seed in args.seeds:
        print(f"[{cid}] evaluating seed {seed}", flush=True)
        r = subprocess.run(py + ["train.py", "--fresh", "--seed", str(seed)],
                           cwd=work, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            return cid, None, None, f"eval failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}"
        per_seed.append(_summarise(work / "runs" / f"seed{seed}" / "results.npz", work, py))

    merged = {}
    for m in per_seed[0]:
        stats = []
        for i in range(len(METRICS)):
            vals = [s[m][i] for s in per_seed]
            mu = sum(vals) / len(vals)
            var = sum((v - mu) ** 2 for v in vals) / len(vals)
            stats.append((mu, var ** 0.5))
        merged[m] = stats
    print(f"[{cid}] done", flush=True)
    return cid, params, merged, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-dir", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--conditions", default=",".join(c[0] for c in CONDITIONS))
    ap.add_argument("--jobs", type=int, default=1,
                    help="Conditions to run concurrently. They are independent.")
    ap.add_argument("--threads", type=int, default=0,
                    help="Torch threads per condition (0 = cores // jobs).")
    ap.add_argument("--skip-tuning", action="store_true",
                    help="Evaluate each condition at the incoming config values.")
    args = ap.parse_args()

    code = Path(args.code_dir).resolve()
    wanted = args.conditions.split(",")
    conds = [c for c in CONDITIONS if c[0] in wanted]
    py = shlex.split(args.python)
    jobs = max(1, min(args.jobs, len(conds)))
    threads = args.threads or max(1, (os.cpu_count() or 4) // jobs)
    print(f"{len(conds)} conditions, {jobs} in parallel, {threads} torch threads each\n",
          flush=True)

    rows, params_by_cond, failures = [], {}, []
    with tempfile.TemporaryDirectory(prefix="cl-fairness-normstudy-") as tmp:
        with cf.ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(_run_condition, cid, norm, mode,
                                   code, tmp, py, args, threads)
                       for cid, norm, mode in conds]
            for fut in cf.as_completed(futures):
                cid, params, merged, err = fut.result()
                if err:
                    failures.append(f"[{cid}] {err}")
                    continue
                params_by_cond[cid] = params
                for m, stats in merged.items():
                    row = {"condition": cid, "method": m, "n_seeds": len(args.seeds)}
                    for i, k in enumerate(METRICS):
                        row[f"{k}_mean"] = f"{stats[i][0]:.4f}"
                        row[f"{k}_std"] = f"{stats[i][1]:.4f}"
                    for pk, pv in (params_by_cond.get(cid) or {}).items():
                        row[pk] = pv
                    rows.append(row)

    for f in failures:
        print("\n" + f)

    if not rows:
        raise SystemExit("every condition failed")

    out_csv = code / "results" / "norm_study.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        fields = (["condition", "method", "n_seeds"]
                  + [f"{k}_{s}" for k in METRICS for s in ("mean", "std")]
                  + TRACKED)
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    by = {(r["condition"], r["method"]): r for r in rows}
    labels = [c[0] for c in conds if c[0] in params_by_cond]
    for metric in ["AccParity", "F-BWT", "BalAcc"]:
        mode = "FIXED config (--skip-tuning)" if args.skip_tuning else "own tuned optimum"
        print(f"\n===== {metric} (each condition at its {mode}) =====")
        print(f"{'method':<14}" + "".join(f"{l:>13}" for l in labels))
        print("-" * (14 + 13 * len(labels)))
        for m in REPORT:
            cells = []
            for l in labels:
                r = by.get((l, m))
                cells.append(f"{float(r[metric+'_mean']):>8.3f}±{float(r[metric+'_std']):<4.3f}"
                             if r else f"{'-':>13}")
            print(f"{m:<14}" + "".join(cells))

    print("\nTuned fairness margins per condition:")
    for cid in labels:
        p = params_by_cond.get(cid, {})
        print(f"  {cid:<11} FAIR-GEM a/b = {p.get('FAIRGEM_ALPHA')}/{p.get('FAIRGEM_BETA')}"
              f"   Fair-DER++ eps a/b = {p.get('FAIRDERPP_EPS_ALPHA')}/{p.get('FAIRDERPP_EPS_BETA')}")
    print(f"\nWrote {out_csv}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
