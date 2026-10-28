"""sensitivity_summary.py — derive sensitivity_summary.csv from the ablations.

The projection-cycle, buffer-size and plain-SGD controls are ordinary
`spurious_shift/run_ablations.py` arms, so their numbers live in
`spurious_shift/results/results_ablations_summary.csv`. This selects those arms
into `spurious_shift/results/sensitivity_summary.csv`.

Usage:
    python analysis/sensitivity_summary.py            # write
    python analysis/sensitivity_summary.py --check    # fail if the CSV differs
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPUR = ROOT / "spurious_shift" / "results"
SRC = SPUR / "results_ablations_summary.csv"
OUT = SPUR / "sensitivity_summary.csv"

# The arms this table reports, in the order the appendix discusses them.
ARMS = [
    "fair_gem_cycles_1", "fair_gem_cycles_3", "fair_gem_cycles_5",
    "fair_gem_cycles_30",
    "agem_mem200", "agem_mem1600",
    "derpp_mem200", "derpp_mem1600",
    "fair_gem_mem200", "fair_gem_mem1600",
    "fair_derpp_mem200", "fair_derpp_mem1600",
    "fair_gem_sgd_proj", "fair_derpp_sgd_proj",
]
FIELDS = ["arm", "n_seeds", "BalAcc_mean", "BalAcc_std",
          "AccParity_mean", "AccParity_std"]


def build() -> str:
    with SRC.open(newline="", encoding="utf-8") as fh:
        src = {r["method"]: r for r in csv.DictReader(fh)}
    missing = [a for a in ARMS if a not in src]
    if missing:
        raise SystemExit(f"arms absent from {SRC.name}: {missing}")
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\n")
    w.writeheader()
    for arm in sorted(ARMS):
        r = src[arm]
        w.writerow({
            "arm": arm,
            "n_seeds": r["n_seeds"],
            "BalAcc_mean": f"{float(r['BalAcc_mean']):.4f}",
            "BalAcc_std": f"{float(r['BalAcc_std']):.4f}",
            "AccParity_mean": f"{float(r['AccParity_mean']):.4f}",
            "AccParity_std": f"{float(r['AccParity_std']):.4f}",
        })
    return buf.getvalue()


def main() -> int:
    text = build()
    if "--check" in sys.argv:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            print(f"STALE {OUT.relative_to(ROOT)} -- run analysis/sensitivity_summary.py")
            return 1
        print("sensitivity_summary.csv is current.")
        return 0
    OUT.write_text(text, encoding="utf-8")
    n = {r.split(",")[1] for r in text.splitlines()[1:]}
    print(f"Wrote {OUT.relative_to(ROOT)} ({len(ARMS)} arms, n_seeds={sorted(n)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
