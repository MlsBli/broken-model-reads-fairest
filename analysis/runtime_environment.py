"""runtime_environment.py — compare method runtime across two environments.

FSW's runtime is dominated by a per-epoch linear program, and the solve time
depends on which backend SciPy resolves `method="highs"` to. Running the
identical configuration in two environments gives FSW wall-clock figures that
differ by roughly two orders of magnitude, while every other method moves only
with general machine speed and FSW's results and sample-retention behaviour are
unchanged. That is a property of the solver stack, not of the method.

This script takes the two runtime tables and writes the comparison to
spurious_shift/results/environment_sensitivity.csv, normalising each
environment by its own DER++ time so machine speed divides out.

Usage:
    python analysis/runtime_environment.py \
        [--env-a <runtime_summary.csv>] [--label-a "..."] \
        [--env-b <runtime_summary.csv>] [--label-b "..."]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "spurious_shift" / "results" / "environment_sensitivity.csv"

ORDER = ["joint_retrain", "finetune", "agem", "der_pp", "group_dro",
         "fairl", "fsw", "fair_gem", "fair_derpp"]
REF = "der_pp"


def read(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as fh:
        return {r["method"]: float(r["total_seconds_mean"])
                for r in csv.DictReader(fh)}


def main() -> int:
    ap = argparse.ArgumentParser()
    # Both sides default to files under spurious_shift/results/, so the
    # comparison is reproducible from the repository alone.
    # runtime_reference_environment.csv is the runtime table from the
    # reference environment.
    ap.add_argument("--env-a",
                    default=str(ROOT / "spurious_shift" / "results"
                               / "runtime_reference_environment.csv"))
    ap.add_argument("--env-b",
                    default=str(ROOT / "spurious_shift" / "results"
                               / "runtime_summary.csv"))
    ap.add_argument("--check", action="store_true",
                    help="verify the committed comparison is current")
    ap.add_argument("--label-a", default="environment A")
    ap.add_argument("--label-b", default="environment B")
    args = ap.parse_args()

    a, b = read(Path(args.env_a)), read(Path(args.env_b))
    rows = []
    for m in ORDER:
        if m not in a or m not in b:
            continue
        # Normalise by each environment's own DER++ so raw machine speed cancels
        # and what remains is method-specific cost.
        ra, rb = a[m] / a[REF], b[m] / b[REF]
        rows.append({
            "method": m,
            "env_a_seconds": f"{a[m]:.3f}",
            "env_b_seconds": f"{b[m]:.3f}",
            "env_a_over_derpp": f"{ra:.4f}",
            "env_b_over_derpp": f"{rb:.4f}",
            "ratio_of_ratios": f"{max(ra, rb) / max(min(ra, rb), 1e-12):.2f}",
        })
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != buf.getvalue():
            print(f"STALE {OUT.relative_to(ROOT)} -- rerun "
                  f"analysis/runtime_environment.py")
            return 1
        print("environment_sensitivity.csv is current.")
        return 0
    OUT.write_text(buf.getvalue(), encoding="utf-8")

    others = [float(r["ratio_of_ratios"]) for r in rows if r["method"] != "fsw"]
    fsw = next(r for r in rows if r["method"] == "fsw")
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"  {args.label_a}: FSW = {fsw['env_a_over_derpp']}x DER++")
    print(f"  {args.label_b}: FSW = {fsw['env_b_over_derpp']}x DER++")
    print(f"  FSW spread {fsw['ratio_of_ratios']}x; "
          f"largest spread among all other methods {max(others):.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
