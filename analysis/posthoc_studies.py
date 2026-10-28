"""posthoc_studies.py — post-hoc studies over the saved results.

None of these studies retrains anything. They read the per-seed matrices and
summary CSVs under <suite>/results/ and write a row-per-finding CSV plus LaTeX
macros to analysis/results/.

  E1  paired permutation tests over seeds (exact, two-sided), with Holm
      correction (E1c)
  E2  equal-accuracy comparison of FSW against the projection methods
  E3  the collapsed-head fairness artifact, as coordinates for a scatter
  E4  accuracy-monitor simulation: when would a utility monitor have fired?
  E5  cost of each unit of disparity reduction
  E6  readout concordance vs. encoder plasticity (frozen vs. fine-tuned arm)

Usage:
    python analysis/posthoc_studies.py            # write posthoc_studies.csv + .tex
    python analysis/posthoc_studies.py --check    # regenerate and diff
"""

from __future__ import annotations

import csv
import difflib
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parent.parent
SPUR = ROOT / "spurious_shift" / "results"
BIOS = ROOT / "bias_in_bios" / "results"
FROZEN = BIOS / "frozen_encoder"
OUT_CSV = ROOT / "analysis" / "results" / "posthoc_studies.csv"
OUT_TEX = ROOT / "analysis" / "results" / "posthoc_studies.tex"

OURS = ("fair_gem", "fair_derpp")


def _final_row_mean(matrix: np.ndarray) -> float:
    m = np.asarray(matrix, dtype=float)
    return float(np.nanmean(m[m.shape[0] - 1, : m.shape[0]]))


def _per_seed(agg, key: str, method: str) -> np.ndarray:
    """Seed-level value of the reported statistic (final-row mean)."""
    stack = np.asarray(agg[f"{key}__{method}__per_seed"], dtype=float)
    return np.array([_final_row_mean(s) for s in stack])


# --------------------------------------------------------------------------
# E1 — exact paired permutation test
# --------------------------------------------------------------------------
def paired_permutation(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    """Exact two-sided paired permutation test on the mean difference.

    Enumerates all 2^n sign flips of the paired differences, so the p-value is
    exact rather than sampled. With n seeds the smallest attainable p-value is
    2/2^n -- report it alongside so the floor is visible.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    observed = float(d.mean())
    count = 0
    for signs in itertools.product([1, -1], repeat=n):
        if abs(float((d * np.array(signs)).mean())) >= abs(observed) - 1e-12:
            count += 1
    return observed, count / (2 ** n), n


def study_e1(rows: list[dict], macros: list[str]) -> None:
    agg = np.load(SPUR / "results_agg.npz", allow_pickle=True)
    # Each of our methods against its own backbone. Fairness gaps are "lower is
    # better".
    pairs = [("fair_gem", "agem"), ("fair_derpp", "der_pp")]
    metrics = [("eo", "AccParity", "down"), ("eo_true", "EO", "down"),
               ("bal_acc", "BalAcc", "up"), ("worst_group", "WorstGroup", "up")]
    for ours, backbone in pairs:
        for key, label, direction in metrics:
            a = _per_seed(agg, key, ours)
            b = _per_seed(agg, key, backbone)
            diff, p, n = paired_permutation(a, b)
            better = (diff < 0) if direction == "down" else (diff > 0)
            rows.append({
                "study": "E1_permutation", "benchmark": "spurious",
                "subject": f"{ours}_vs_{backbone}", "metric": label,
                "value": f"{diff:.4f}", "p_value": f"{p:.4f}",
                "n": n, "note": f"floor={2 / 2 ** n:.4f}; "
                                f"{'favours ours' if better else 'favours backbone'}",
            })
            tag = "Fairgem" if ours == "fair_gem" else "Fairderpp"
            macros.append(
                f"\\newcommand{{\\Perm{tag}{label}P}}{{{p:.3f}}}")
            macros.append(
                f"\\newcommand{{\\Perm{tag}{label}D}}{{{abs(diff):.3f}}}")
    # FAIR-GEM-SGD against FSW, paired over the same seeds (the SGD arm from
    # results_ablations_agg.npz, FSW from the main aggregate).
    abl_path = SPUR / "results_ablations_agg.npz"
    if abl_path.exists():
        abl = np.load(abl_path, allow_pickle=True)
        key = "eo"          # on Spurious this array is the accuracy-parity gap
        if f"{key}__fair_gem_sgd_proj__per_seed" in abl.files:
            a = _per_seed(abl, key, "fair_gem_sgd_proj")
            b = _per_seed(agg, key, "fsw")
            diff, p, n = paired_permutation(a, b)
            rows.append({
                "study": "E1_permutation", "benchmark": "spurious",
                "subject": "fair_gem_sgd_vs_fsw", "metric": "AccParity",
                "value": f"{diff:.4f}", "p_value": f"{p:.4f}",
                "n": n, "note": f"floor={2 / 2 ** n:.4f}; "
                                f"{'indistinguishable' if p >= 0.05 else 'distinguishable'}"
                                f" at 0.05",
            })
            macros.append(f"\\newcommand{{\\PermSgdFswAPP}}{{{p:.3f}}}")
            macros.append(f"\\newcommand{{\\PermSgdFswAPD}}{{{abs(diff):.3f}}}")

    # The attainable floor depends on the seed budget, so derive it from the
    # number of seeds in the aggregate.
    n_used = len(_per_seed(agg, "eo", "der_pp"))
    macros.append(f"\\newcommand{{\\PermNSeeds}}{{{n_used}}}")
    macros.append(f"\\newcommand{{\\PermFloor}}{{{2 / 2 ** n_used:.4f}}}")
    # Floors at seed budgets common in this literature, which cannot reach
    # conventional significance however large the effect.
    for n, tag in ((3, "Three"), (5, "Five")):
        macros.append(f"\\newcommand{{\\PermFloor{tag}}}{{{2 / 2 ** n:.3f}}}")
    # Bound on how much the effect estimates move between the first five seeds
    # and the full seed set.
    if n_used > 5:
        shifts = []
        for ours, backbone in pairs:
            a5 = _per_seed(agg, "eo", ours)[:5] - _per_seed(agg, "eo", backbone)[:5]
            aN = _per_seed(agg, "eo", ours) - _per_seed(agg, "eo", backbone)
            shifts.append(abs(float(aN.mean()) - float(a5.mean())))
        # Round up to a clean bound so the prose reads as a bound, not a value.
        import math
        bound = math.ceil(max(shifts) * 1000) / 1000
        macros.append(f"\\newcommand{{\\PermEffectShift}}{{{bound:.3f}}}")


# --------------------------------------------------------------------------
# E1b - the same exact permutation test on Bias in Bios
#
# The floor 2/2^n applies here too: at n=3 it is 0.250, so no comparison could
# reach p<0.05 however large the effect.
#
# The npz metric KEY differs between benchmarks: Spurious Shift stores accuracy
# parity under `eo` and equalized odds under `eo_true`, while BIOS stores
# equalized odds under `eo` and accuracy parity under `acc_parity`. The key list
# is therefore written per benchmark -- sharing one would silently relabel every
# BIOS row as the wrong metric.
# --------------------------------------------------------------------------
def study_e1_bios(rows: list[dict], macros: list[str]) -> None:
    path = BIOS / "results_agg.npz"
    if not path.exists():
        return
    agg = np.load(path, allow_pickle=True)
    if "bal_acc__agem__per_seed" not in agg.files:
        # Aggregate without per-seed arrays (see bias_in_bios/aggregate.py).
        # Emit nothing rather than a wrong number.
        return
    pairs = [("fair_gem", "agem"), ("fair_derpp", "derpp")]
    metrics = [("acc_parity", "AccParity", "down"), ("eo", "EO", "down"),
               ("bal_acc", "BalAcc", "up"), ("worst_group", "WorstGroup", "up")]
    for ours, backbone in pairs:
        for key, label, direction in metrics:
            if f"{key}__{ours}__per_seed" not in agg.files:
                continue
            a = _per_seed(agg, key, ours)
            b = _per_seed(agg, key, backbone)
            diff, p, n = paired_permutation(a, b)
            better = (diff < 0) if direction == "down" else (diff > 0)
            rows.append({
                "study": "E1_permutation", "benchmark": "bios",
                "subject": f"{ours}_vs_{backbone}", "metric": label,
                "value": f"{diff:.4f}", "p_value": f"{p:.4f}",
                "n": n, "note": f"floor={2 / 2 ** n:.4f}; "
                                f"{'favours ours' if better else 'favours backbone'}",
            })
            tag = "Fairgem" if ours == "fair_gem" else "Fairderpp"
            macros.append(f"\\newcommand{{\\PermBios{tag}{label}P}}{{{p:.3f}}}")
            macros.append(f"\\newcommand{{\\PermBios{tag}{label}D}}{{{abs(diff):.3f}}}")
    n_used = len(_per_seed(agg, "bal_acc", "agem"))
    macros.append(f"\\newcommand{{\\PermNSeedsBios}}{{{n_used}}}")
    macros.append(f"\\newcommand{{\\PermFloorBios}}{{{2 / 2 ** n_used:.4f}}}")


# --------------------------------------------------------------------------
# E1c - multiplicity
#
# E1 and E1b together run a family of exact tests, so their p-values are
# corrected for the number of comparisons.
#
# Holm-Bonferroni rather than plain Bonferroni: it controls the same family-wise
# error rate, is uniformly at least as powerful, and needs no independence
# assumption -- which matters here, because these metrics are computed from the
# same predictions on the same seeds and are strongly dependent. Under
# dependence Holm is conservative.
#
# Two families are reported: per benchmark (one set of hypotheses about the
# same methods on the same sequence), and pooled over every test, the most
# conservative reading.
#
# The smallest attainable *corrected* p-value is m * 2/2^n for m tests at n
# paired seeds, so the seed budget a design needs grows with the number of
# comparisons it makes.
# --------------------------------------------------------------------------
def _holm(ps: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, in the input order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    m = len(ps)
    out = [0.0] * len(ps)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[idx]))
        out[idx] = running
    return out


def study_e1_multiplicity(rows: list[dict], macros: list[str]) -> None:
    perm = [r for r in rows if r["study"] == "E1_permutation"]
    if not perm:
        return

    pooled_adj = _holm([float(r["p_value"]) for r in perm])
    per_family: dict[str, list[int]] = {}
    for i, r in enumerate(perm):
        per_family.setdefault(r["benchmark"], []).append(i)

    family_adj = [0.0] * len(perm)
    for bench, idxs in per_family.items():
        adj = _holm([float(perm[i]["p_value"]) for i in idxs])
        for slot, i in enumerate(idxs):
            family_adj[i] = adj[slot]

    for i, r in enumerate(perm):
        r["p_holm_family"] = f"{family_adj[i]:.4f}"
        r["p_holm_pooled"] = f"{pooled_adj[i]:.4f}"

    macros.append("% --- E1c: multiplicity over the exact-permutation family ---")
    macros.append(f"\\newcommand{{\\PermNTests}}{{{len(perm)}}}")
    for bench, idxs in sorted(per_family.items()):
        tag = "Spurious" if bench == "spurious" else "Bios"
        macros.append(f"\\newcommand{{\\PermNTests{tag}}}{{{len(idxs)}}}")
        # The corrected floor: the smallest possible p-value once the family
        # size is accounted for.
        n = int(perm[idxs[0]]["n"])
        macros.append(
            f"\\newcommand{{\\PermFloorHolm{tag}}}"
            f"{{{min(1.0, len(idxs) * 2 / 2 ** n):.3f}}}")
        # The same at seed budgets common in this literature, where the
        # corrected floor can reach 1.
        for nb, nbtag in ((3, "Three"), (5, "Five")):
            macros.append(
                f"\\newcommand{{\\PermFloorHolm{tag}{nbtag}}}"
                f"{{{min(1.0, len(idxs) * 2 / 2 ** nb):.2f}}}")

    # Individual adjusted values for selected comparisons.
    named = {
        ("spurious", "fair_derpp_vs_der_pp", "AccParity"): "FairderppAccParity",
        ("spurious", "fair_gem_vs_agem", "AccParity"): "FairgemAccParity",
        ("bios", "fair_derpp_vs_derpp", "EO"): "BiosFairderppEO",
    }
    for i, r in enumerate(perm):
        key = (r["benchmark"], r["subject"], r["metric"])
        if key in named:
            stem = named[key]
            macros.append(
                f"\\newcommand{{\\Perm{stem}Holm}}{{{family_adj[i]:.3f}}}")
            macros.append(
                f"\\newcommand{{\\Perm{stem}HolmPooled}}{{{pooled_adj[i]:.3f}}}")

    # How many of the family clear 0.05 before and after correction.
    for bench, idxs in sorted(per_family.items()):
        tag = "Spurious" if bench == "spurious" else "Bios"
        raw = sum(1 for i in idxs if float(perm[i]["p_value"]) < 0.05)
        adj = sum(1 for i in idxs if family_adj[i] < 0.05)
        macros.append(f"\\newcommand{{\\PermNSig{tag}}}{{{raw}}}")
        macros.append(f"\\newcommand{{\\PermNSigHolm{tag}}}{{{adj}}}")
    macros.append("")


# --------------------------------------------------------------------------
# Realized-step rates: how often the projected constraint survives the optimizer
# --------------------------------------------------------------------------
def study_realized(rows: list[dict], macros: list[str]) -> None:
    """Fraction of sampled updates where the applied step keeps the half-space.

    Reported in the paper as a range across methods. Reads
    analysis/results/mechanism_diagnostics.csv (written by diagnostics.py).
    """
    path = ROOT / "analysis" / "results" / "mechanism_diagnostics.csv"
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as fh:
        data = [r for r in csv.DictReader(fh)
                if r.get("benchmark") == "spurious" and r.get("realized_safety_rate")]
    if not data:
        return
    per_method = {}
    for r in data:
        per_method.setdefault(r["method"], []).append(float(r["realized_safety_rate"]))
    means = {m: float(np.mean(v)) for m, v in per_method.items()}
    for m, v in means.items():
        tag = "Fairgem" if m == "fair_gem" else "Fairderpp" if m == "fair_derpp" else m
        macros.append(f"\\newcommand{{\\Realized{tag}Pct}}{{{v * 100:.0f}}}")
        rows.append({
            "study": "realized_step", "benchmark": "spurious", "subject": m,
            "metric": "realized_safety_rate (method-level mean)",
            "value": f"{v:.4f}", "p_value": "", "n": len(per_method[m]),
            "note": "fraction of sampled updates whose applied step keeps the half-space",
        })
    macros.append(f"\\newcommand{{\\RealizedLoPct}}{{{min(means.values()) * 100:.0f}}}")
    macros.append(f"\\newcommand{{\\RealizedHiPct}}{{{max(means.values()) * 100:.0f}}}")


# --------------------------------------------------------------------------
# E2 — equal-accuracy comparison against FSW
# --------------------------------------------------------------------------
def study_e2(rows: list[dict], macros: list[str]) -> None:
    curves: dict[str, list[tuple[float, float, float]]] = {}
    with (SPUR / "tradeoff_curves_agg.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            curves.setdefault(row["method"], []).append(
                (float(row["bal_mean"]), float(row["acc_parity_mean"]),
                 float(row["scale"])))
    for method in curves:
        curves[method].sort()

    def gap_at(method: str, target: float) -> float | None:
        """Linear interpolation of the disparity gap at a target accuracy."""
        pts = curves[method]
        xs = [p[0] for p in pts]
        if not (min(xs) <= target <= max(xs)):
            return None
        return float(np.interp(target, xs, [p[1] for p in pts]))

    # Compare only on the accuracy band all three methods reach, and report
    # when there is none. If their balanced-accuracy ranges are disjoint, the
    # intersection is empty and the targets fall back to the union band, where
    # at most one method has a value at each -- no comparison is possible. The
    # band status is written as a row of its own so this is visible in the CSV.
    TRIPLE = ("fsw", "fair_derpp", "fair_gem")
    lows = {m: min(p[0] for p in curves[m]) for m in TRIPLE}
    highs = {m: max(p[0] for p in curves[m]) for m in TRIPLE}
    lo, hi = max(lows.values()), min(highs.values())
    common = lo <= hi
    rows.append({
        "study": "E2_equal_accuracy", "benchmark": "spurious",
        "subject": "common_band", "metric": "BalAcc reachable by all three",
        "value": ("" if common else "EMPTY") or f"[{lo:.4f},{hi:.4f}]",
        "p_value": "", "n": 5,
        "note": "; ".join(f"{m} reaches [{lows[m]:.4f},{highs[m]:.4f}]"
                          for m in TRIPLE)
        + ("" if common else
           " -- the intersection is empty, so no accuracy target is reachable "
           "by all three and the targets below use the union band; at most one "
           "method has a value at each"),
    })
    macros.append(f"\\newcommand{{\\EqAccCommonBand}}{{{'yes' if common else 'no'}}}")
    for m, tag in (("fsw", "Fsw"), ("fair_gem", "Fairgem"),
                   ("fair_derpp", "Fairderpp")):
        macros.append(f"\\newcommand{{\\EqAccRangeLo{tag}}}{{{lows[m]:.3f}}}")
        macros.append(f"\\newcommand{{\\EqAccRangeHi{tag}}}{{{highs[m]:.3f}}}")
    if not common:
        lo, hi = min(lows.values()), max(highs.values())
    for target in np.linspace(lo, hi, 5):
        entry = {"study": "E2_equal_accuracy", "benchmark": "spurious",
                 "subject": f"balacc={target:.3f}", "metric": "AccParity",
                 "value": "", "p_value": "", "n": 5,
                 "note": "" if common else "union-band target; not a comparison"}
        got = {m: gap_at(m, float(target)) for m in ("fsw", "fair_gem", "fair_derpp")}
        entry["value"] = "; ".join(
            f"{m}={v:.4f}" for m, v in got.items() if v is not None)
        rows.append(entry)
    # Headline: the overlap band and each method's gap at its midpoint.
    mid = float((lo + hi) / 2)
    macros.append(f"\\newcommand{{\\EqAccLo}}{{{lo:.3f}}}")
    macros.append(f"\\newcommand{{\\EqAccHi}}{{{hi:.3f}}}")
    macros.append(f"\\newcommand{{\\EqAccMid}}{{{mid:.3f}}}")
    for m, tag in (("fsw", "Fsw"), ("fair_gem", "Fairgem"),
                   ("fair_derpp", "Fairderpp")):
        v = gap_at(m, mid)
        if v is not None:
            macros.append(f"\\newcommand{{\\EqAcc{tag}}}{{{v:.3f}}}")


# --------------------------------------------------------------------------
# E3 — the collapsed-head fairness artifact
# --------------------------------------------------------------------------
def study_e3(rows: list[dict], macros: list[str]) -> None:
    with (BIOS / "results_test_summary.csv").open(encoding="utf-8") as fh:
        data = {r["method"]: r for r in csv.DictReader(fh)}
    collapsed, healthy = [], []
    for method, row in data.items():
        acc, eo = float(row["BalAcc_mean"]), float(row["EO_mean"])
        bucket = "collapsed" if acc < 0.30 else ("healthy" if acc > 0.50 else "intermediate")
        rows.append({
            "study": "E3_collapse_artifact", "benchmark": "bios",
            "subject": method, "metric": "BalAcc_head;EO_head",
            # n is the seed count the mean was taken over, read from the
            # source summary.
            "value": f"{acc:.4f};{eo:.4f}", "p_value": "",
            "n": int(row["n_seeds"]),
            "note": bucket,
        })
        (collapsed if bucket == "collapsed" else
         healthy if bucket == "healthy" else []).append((method, acc, eo))
    # The lowest gaps belong to the broken models. The comparison point is the
    # functioning model with the lowest gap (not the most accurate one), the
    # strongest case the artifact has to beat.
    worst_acc = min(healthy, key=lambda t: t[2])
    macros.append(f"\\newcommand{{\\CollapseNCollapsed}}{{{len(collapsed)}}}")
    # Word form as well, for prose of the form "N of nine methods".
    _words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
              7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    macros.append(f"\\newcommand{{\\CollapseNCollapsedW}}"
                  f"{{{_words.get(len(collapsed), str(len(collapsed)))}}}")
    macros.append(
        f"\\newcommand{{\\CollapseMaxEO}}{{{max(e for _, _, e in collapsed):.3f}}}")
    macros.append(
        f"\\newcommand{{\\CollapseMaxAcc}}{{{max(a for _, a, _ in collapsed):.3f}}}")
    macros.append(f"\\newcommand{{\\CollapseLowestGapHealthyEO}}{{{worst_acc[2]:.3f}}}")
    macros.append(
        f"\\newcommand{{\\CollapseLowestGapHealthyAcc}}{{{worst_acc[1]:.3f}}}")


# --------------------------------------------------------------------------
# E4 — accuracy-monitor simulation
# --------------------------------------------------------------------------
def study_e4(rows: list[dict], macros: list[str]) -> None:
    """Retention monitors: does a disparity alarm fire before a utility alarm?

    A deployment monitor does not watch average accuracy over a growing task
    set -- that legitimately falls when harder tasks arrive, and on this
    benchmark it drops for every method at the task-3 rotation. What it watches
    is *retention*: performance on what the model was already good at. So the
    utility monitor tracks the mean drop on past tasks relative to when each was
    learned, and alarms when that exceeds `tol`. The fairness monitor tracks the
    mean accuracy-parity gap on those same past tasks and alarms at `gap_tol`.

    Thresholds are conventions, so both are swept and we report the ordering --
    which alarm fires first -- which is the threshold-robust statement.
    """
    agg = np.load(SPUR / "results_agg.npz", allow_pickle=True)
    methods = [str(m) for m in agg["methods"]]
    acc_tols = [0.02, 0.05, 0.10]
    gap_tols = [0.10, 0.15, 0.20]

    def first_alarm(series, threshold, rising: bool):
        for t, v in enumerate(series):
            if v is None or np.isnan(v):
                continue
            if (v > threshold) if rising else (v < -threshold):
                return t
        return None

    for method in methods:
        acc = np.asarray(agg[f"bal_acc__{method}__per_seed"], dtype=float)
        gap = np.asarray(agg[f"eo__{method}__per_seed"], dtype=float)
        n_seeds, T, _ = acc.shape
        for atol in acc_tols:
            for gtol in gap_tols:
                a_fire, g_fire, gap_first = [], [], []
                for s in range(n_seeds):
                    # retention drop on past tasks, and their disparity
                    drop, pgap = [], []
                    for t in range(T):
                        past = [acc[s, t, i] - acc[s, i, i] for i in range(t)]
                        pg = [gap[s, t, i] for i in range(t)]
                        drop.append(float(np.nanmean(past)) if past else np.nan)
                        pgap.append(float(np.nanmean(pg)) if pg else np.nan)
                    af = first_alarm(drop, atol, rising=False)
                    gf = first_alarm(pgap, gtol, rising=True)
                    a_fire.append(np.nan if af is None else af)
                    g_fire.append(np.nan if gf is None else gf)
                    if gf is not None and (af is None or gf < af):
                        gap_first.append(1)
                    else:
                        gap_first.append(0)
                rows.append({
                    "study": "E4_monitor", "benchmark": "spurious",
                    "subject": method,
                    "metric": f"acc_tol={atol};gap_tol={gtol}",
                    "value": f"acc_alarm={np.nanmean(a_fire):.2f};"
                             f"gap_alarm={np.nanmean(g_fire):.2f};"
                             f"gap_first={sum(gap_first)}/{n_seeds}",
                    "p_value": "", "n": n_seeds,
                    "note": f"acc silent on {int(np.sum(np.isnan(a_fire)))}/{n_seeds}; "
                            f"gap silent on {int(np.sum(np.isnan(g_fire)))}/{n_seeds}",
                })

    # Headline at a 10% retention tolerance -- the regime in which a deployment
    # would judge the model to be retaining acceptably, and therefore the regime
    # in which a silent utility monitor is the interesting result. At tighter
    # tolerances the task-3 feature rotation trips both alarms for most methods,
    # which we state rather than tune around.
    atol, gtol = 0.10, 0.15
    for method, tag in (("der_pp", "Derpp"), ("fair_derpp", "Fairderpp"),
                        ("fsw", "Fsw"), ("fair_gem", "Fairgem"),
                        ("agem", "Agem")):
        acc = np.asarray(agg[f"bal_acc__{method}__per_seed"], dtype=float)
        gap = np.asarray(agg[f"eo__{method}__per_seed"], dtype=float)
        n_seeds, T, _ = acc.shape
        a_silent = g_fired = 0
        fire_tasks = []
        for s in range(n_seeds):
            drop, pgap = [], []
            for t in range(T):
                past = [acc[s, t, i] - acc[s, i, i] for i in range(t)]
                pg = [gap[s, t, i] for i in range(t)]
                drop.append(float(np.nanmean(past)) if past else np.nan)
                pgap.append(float(np.nanmean(pg)) if pg else np.nan)
            if first_alarm(drop, atol, rising=False) is None:
                a_silent += 1
            gf = first_alarm(pgap, gtol, rising=True)
            if gf is not None:
                g_fired += 1
                fire_tasks.append(gf)
        macros.append(f"\\newcommand{{\\Mon{tag}AccSilent}}{{{a_silent}}}")
        macros.append(f"\\newcommand{{\\Mon{tag}GapFired}}{{{g_fired}}}")
        if fire_tasks:
            macros.append(
                f"\\newcommand{{\\Mon{tag}GapTask}}{{{np.mean(fire_tasks):.1f}}}")
    macros.append(f"\\newcommand{{\\MonSeeds}}{{{n_seeds}}}")
    macros.append(f"\\newcommand{{\\MonAccTol}}{{{int(atol * 100)}}}")
    macros.append(f"\\newcommand{{\\MonGapTol}}{{{gtol:.2f}}}")
    macros.append(f"\\newcommand{{\\MonAccTolTight}}{{{2}}}")


# --------------------------------------------------------------------------
# E5 — cost per unit of disparity reduction
# --------------------------------------------------------------------------
def study_e5(rows: list[dict], macros: list[str]) -> None:
    for bench, code, ref in (("spurious", SPUR, "der_pp"), ("bios", BIOS, "derpp")):
        with (code / "runtime_summary.csv").open(encoding="utf-8") as fh:
            rt = {r["method"]: r for r in csv.DictReader(fh)}
        summary_name = ("results_summary.csv" if bench == "spurious"
                        else "results_test_summary.csv")
        gap_col = "AccParity_mean"
        with (code / summary_name).open(encoding="utf-8") as fh:
            res = {r["method"]: r for r in csv.DictReader(fh)}
        base_gap = float(res[ref][gap_col])
        base_time = float(rt[ref]["total_seconds_mean"])
        for method in rt:
            if method not in res or method == ref:
                continue
            gap = float(res[method][gap_col])
            reduction = base_gap - gap
            extra = float(rt[method]["total_seconds_mean"]) - base_time
            rows.append({
                "study": "E5_cost", "benchmark": bench, "subject": method,
                "metric": "gap_reduction_vs_replay_backbone;extra_seconds;seconds_per_point",
                "value": f"{reduction:.4f};{extra:.1f};"
                         f"{(extra / (reduction * 100)) if reduction > 1e-6 else float('nan'):.1f}",
                "p_value": "", "n": int(rt[method]["n_seeds"]),
                "note": f"relative to {ref}",
            })


def _avg_ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks, so exact ties do not bias the correlation either way."""
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    for v in np.unique(x):
        tie = x == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return ranks


def _spearman_exact(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    """Spearman rho with an EXACT two-sided permutation p-value.

    Nine methods means 9! = 362880 rank orderings, so the null is enumerable and
    the asymptotic approximation is unnecessary -- the same reason study_e1
    enumerates sign flips instead of trusting a t-approximation at these sizes.
    Pearson-on-average-ranks rather than the 1-6*sum(d^2) shortcut, which is only
    valid without ties.
    """
    ra, rb = _avg_ranks(a), _avg_ranks(b)
    rho = float(np.corrcoef(ra, rb)[0, 1])
    ra_c = ra - ra.mean()
    denom = float(np.sqrt((ra_c ** 2).sum() * ((rb - rb.mean()) ** 2).sum()))
    perms = np.array(list(itertools.permutations(rb)), dtype=float)
    null = ((perms - perms.mean(axis=1, keepdims=True)) @ ra_c) / denom
    p = float((np.abs(null) >= abs(rho) - 1e-12).mean())
    return rho, p, len(null)


def study_e6(rows: list[dict], macros: list[str]) -> None:
    """Does the readout disagreement survive freezing the encoder?

    Memory-refit and shared-data NCM rank the methods differently. Freezing the
    encoder removes end-to-end plasticity and changes nothing else, so comparing
    readout concordance across the two arms tests whether plasticity causes the
    disagreement.
    """
    arms = (("finetuned", "Ft", BIOS), ("frozen", "Fz", FROZEN))
    for _, _, code in arms:
        if not ((code / "results_ncm_full_summary.csv").exists()
                and (code / "results_retune_summary.csv").exists()):
            return          # emit nothing rather than half a comparison
    for arm, tag, code in arms:
        with (code / "results_ncm_full_summary.csv").open(encoding="utf-8") as fh:
            ncm = {r["method"]: float(r["BalAcc_mean"]) for r in csv.DictReader(fh)}
        with (code / "results_retune_summary.csv").open(encoding="utf-8") as fh:
            mem = {r["method"]: float(r["BalAcc_mean"]) for r in csv.DictReader(fh)}
        shared = sorted(set(ncm) & set(mem))
        rho, pv, n_null = _spearman_exact(
            np.array([ncm[m] for m in shared]), np.array([mem[m] for m in shared]))
        floor = 1.0 / n_null
        pstr = f"$<${floor:.5f}" if pv < floor else f"{pv:.4f}"
        macros.append(f"\\newcommand{{\\RhoReadout{tag}}}{{{rho:.3f}}}")
        macros.append(f"\\newcommand{{\\RhoReadout{tag}P}}{{{pstr}}}")
        macros.append(f"\\newcommand{{\\RhoReadout{tag}N}}{{{len(shared)}}}")
        rows.append({
            "study": "E6_readout_concordance", "benchmark": "bios",
            "subject": f"{arm}_encoder",
            "metric": "spearman_rho__ncm_vs_memory_refit",
            "value": f"{rho:.4f}", "p_value": f"{pv:.6f}", "n": len(shared),
            "note": f"exact two-sided permutation over {n_null} rank orderings",
        })
    macros.append("")


def build() -> tuple[str, str]:
    rows: list[dict] = []
    macros: list[str] = [
        "% analysis/results/posthoc_studies.tex",
        "% Generated by analysis/posthoc_studies.py.",
        "",
    ]
    study_e1(rows, macros)
    study_e1_bios(rows, macros)
    # Must run after both permutation studies: it corrects across their union.
    study_e1_multiplicity(rows, macros)
    study_realized(rows, macros)
    study_e2(rows, macros)
    study_e3(rows, macros)
    study_e4(rows, macros)
    study_e5(rows, macros)
    study_e6(rows, macros)

    fields = ["study", "benchmark", "subject", "metric", "value", "p_value",
              "p_holm_family", "p_holm_pooled", "n", "note"]
    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue(), "\n".join(macros) + "\n"


def main() -> int:
    csv_text, tex_text = build()
    if "--check" in sys.argv:
        bad = False
        for path, text in ((OUT_CSV, csv_text), (OUT_TEX, tex_text)):
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                print(f"STALE {path.relative_to(ROOT)}")
                if path.exists():
                    for line in list(difflib.unified_diff(
                            path.read_text(encoding="utf-8").splitlines(), text.splitlines(),
                            lineterm="", n=0))[:20]:
                        print(line)
                bad = True
        if not bad:
            print("post-hoc studies are current.")
        return 1 if bad else 0
    OUT_CSV.write_text(csv_text, encoding="utf-8")
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(tex_text, encoding="utf-8")
    n = sum(1 for line in tex_text.splitlines() if line.startswith("\\newcommand"))
    print(f"Wrote {OUT_CSV.name} ({csv_text.count(chr(10)) - 1} rows) "
          f"and {OUT_TEX.relative_to(ROOT)} ({n} macros).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
