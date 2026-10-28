"""measure_truncation.py — how much of a Bias-in-Bios biography survives tokenization.

The Bias-in-Bios pipeline tokenizes with `max_length=32, truncation=True,
add_special_tokens=True` (`bias_in_bios/datasets/bios_tasks.py::_tokenize`),
matching the FaIRL reference loader. This measures how much of each biography
that budget discards.

The measurement needs no GPU and no training run: it streams biographies from the
Hugging Face dataset and tokenizes them with the same tokenizer the pipeline
uses, once with truncation off, and reports the length distribution against
candidate budgets.

It reports the untruncated distribution, not a model outcome. Whether a larger
budget changes head collapse is a separate question that needs retraining.

Usage:
    python analysis/measure_truncation.py            # measure and write the CSV
    python analysis/measure_truncation.py --check    # fail if the CSV differs
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bias_in_bios" / "results" / "truncation_summary.csv"

DATASET = "LabHC/bias_in_bios"
TOKENIZER = "bert-base-uncased"
N_SAMPLE = 20000
BUDGETS = (32, 64, 128)
CONFIGURED = 32   # config.BIOS_BERT_MAXLEN


def measure(n: int = N_SAMPLE) -> list[dict]:
    import numpy as np
    from datasets import load_dataset
    from transformers import BertTokenizerFast

    ds = load_dataset(DATASET, split="train", streaming=True)
    texts = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        texts.append(row.get("hard_text") or row.get("text") or "")

    tok = BertTokenizerFast.from_pretrained(TOKENIZER)
    # add_special_tokens=True matches _tokenize, so [CLS]/[SEP] count against
    # the budget: a budget of 32 leaves 30 content wordpieces.
    lens = np.array([len(tok.encode(t, add_special_tokens=True, truncation=False))
                     for t in texts])

    rows = []
    for budget in BUDGETS:
        kept = np.minimum(lens, budget)
        rows.append({
            "dataset": DATASET,
            "split": "train",
            "n_sampled": len(lens),
            "max_length": budget,
            "configured": int(budget == CONFIGURED),
            "mean_tokens_untruncated": f"{lens.mean():.1f}",
            "median_tokens_untruncated": f"{np.median(lens):.0f}",
            "p90_tokens_untruncated": f"{np.percentile(lens, 90):.0f}",
            "max_tokens_untruncated": int(lens.max()),
            "pct_bios_truncated": f"{100 * (lens > budget).mean():.1f}",
            "pct_wordpieces_kept": f"{100 * kept.sum() / lens.sum():.1f}",
        })
    return rows


def write(rows: list[dict]) -> None:
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="re-measure and fail if the tracked CSV differs")
    ap.add_argument("-n", type=int, default=N_SAMPLE)
    args = ap.parse_args()

    try:
        rows = measure(args.n)
    except Exception as exc:                      # network or dataset unavailable
        print(f"measurement unavailable ({type(exc).__name__}: {exc})")
        if args.check and OUT.exists():
            print(f"keeping the tracked {OUT.name}; it records the measured values")
            return 0
        return 1

    if args.check:
        if not OUT.exists():
            print(f"{OUT.name} missing")
            return 1
        old = list(csv.DictReader(OUT.open(encoding="utf-8")))
        # Streaming order is fixed, so the same n reproduces the same statistics.
        if [dict(r) for r in old] != rows:
            print(f"{OUT.name} is stale; re-run without --check")
            return 1
        print(f"{OUT.name} is current")
        return 0

    write(rows)
    cfg = next(r for r in rows if r["configured"] == 1)
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"  {cfg['n_sampled']} bios sampled; mean {cfg['mean_tokens_untruncated']} "
          f"wordpieces untruncated")
    print(f"  at max_length={cfg['max_length']}: {cfg['pct_bios_truncated']}% of bios "
          f"truncated, {cfg['pct_wordpieces_kept']}% of wordpieces kept")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
