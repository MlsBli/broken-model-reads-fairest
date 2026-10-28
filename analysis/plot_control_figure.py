r"""plot_control_figure.py -- Figure 3 (control_gap_vs_joint.pdf).

Two panels of accuracy-parity gap per method, each against joint retraining's
level. On Bias in Bios (head refit on a shared pool) every continual learner
sits at joint retraining's gap, so the sequence adds no disparity for a
constraint to remove. On Spurious Shift DER++ ends above joint retraining and
the constrained updates cut its gap. Reads
bias_in_bios/results/results_probe_shared_summary.csv and
spurious_shift/results/results_summary.csv.

Usage:
  python analysis/plot_control_figure.py
"""
from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parent.parent
SPUR = ROOT / "spurious_shift"
BIOS = ROOT / "bias_in_bios"
OUT = ROOT / "analysis" / "results" / "figures"

sys.path.insert(0, str(SPUR))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import plot_style as ps  # noqa: E402

ORDER = ["joint_retrain", "finetune", "agem", "derpp", "group_dro", "fairl",
         "fsw", "fair_gem", "fair_derpp"]
ALIASES = {"der_pp": "derpp"}


def _read(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return {ALIASES.get(r["method"], r["method"]): r for r in csv.DictReader(fh)}


def _panel(ax, rows, title):
    ys = [float(rows[m]["AccParity_mean"]) for m in ORDER]
    sd = [float(rows[m]["AccParity_std"]) for m in ORDER]
    xs = range(len(ORDER))
    for x, m, y, s in zip(xs, ORDER, ys, sd):
        ours = m in ps.OURS
        ax.bar(x, y, width=0.72, color=ps.METHOD_COLORS.get(m, "0.5"),
               alpha=0.9 if ours else 0.75, hatch="////" if ours else None,
               edgecolor="white" if ours else "none", lw=0, zorder=2)
        ax.errorbar(x, y, yerr=s, fmt="none", ecolor="0.2", elinewidth=0.7,
                    capsize=1.6, zorder=3)
    ax.axhline(ys[0], color="0.35", ls=(0, (4, 2)), lw=0.9, zorder=1)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([ps.display_name(m) for m in ORDER], rotation=40,
                       ha="right", rotation_mode="anchor", fontsize=6.8)
    ax.set_title(title, fontsize=7.6)
    ax.set_ylabel(ps.D_AP + r" $\downarrow$")
    ax.grid(True, axis="y", alpha=0.18, lw=0.5)
    ax.set_axisbelow(True)
    return ys


def main() -> Path:
    ps.set_style()
    bios = _read(BIOS / "results" / "results_probe_shared_summary.csv")
    spur = _read(SPUR / "results" / "results_summary.csv")
    fig, (ax_b, ax_s) = plt.subplots(1, 2, figsize=(ps.FULL_W, 2.15), sharey=True)
    _panel(ax_b, bios, "(a) Bias in Bios, refit head")
    ys = _panel(ax_s, spur, "(b) Spurious Shift")
    ax_s.set_ylabel("")
    top = max(float(r["AccParity_mean"]) + float(r["AccParity_std"]) for r in spur.values())
    ax_b.set_ylim(0, top * 1.12)
    gi = ORDER.index("group_dro")
    ax_s.annotate("acc.\ncollapsed", (gi, ys[gi]), xytext=(0, 4),
                  textcoords="offset points", ha="center", va="bottom",
                  fontsize=6.2, color="0.3")
    fig.tight_layout(pad=0.35, w_pad=1.6)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "control_gap_vs_joint.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(f"wrote {main().relative_to(ROOT)}")
