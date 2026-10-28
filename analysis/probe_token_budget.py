"""probe_token_budget.py — what the 32-token budget costs the representation.

`measure_truncation.py` says how much text the budget discards. This says how
much *information* that costs, to the extent that can be measured without
retraining.

Method: freeze pretrained `bert-base-uncased`, embed the same biographies at
several `max_length` values, and fit a linear probe on the pooled output for two
targets -- profession (25 classes, the task) and gender (the protected
attribute). The gap between budgets bounds how much profession and gender
information the truncation removes from the encoder's input.

What this does NOT establish: it probes the *frozen pretrained* encoder, not
the continually fine-tuned one, and says nothing about whether head collapse
softens at a larger budget, which needs retraining. It is a bound on input
information, not a substitute for that experiment.

Single split, no seed replication, so treat the numbers as observational rather
than as a significance claim.

Usage:
    python analysis/probe_token_budget.py                 # measure, write the CSV
    python analysis/probe_token_budget.py --budgets 32 64 # subset
"""

from __future__ import annotations

import argparse
import csv
import collections
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bias_in_bios" / "results" / "token_budget_probe.csv"

DATASET = "LabHC/bias_in_bios"
ENCODER = "bert-base-uncased"
N_CLASSES = 25          # matches config.N_CLASSES / the task construction
N_TRAIN, N_TEST = 9000, 3000
BUDGETS = (32, 64, 128)


def run(budgets=BUDGETS, n_train=N_TRAIN, n_test=N_TEST) -> list[dict]:
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import BertTokenizerFast, BertModel
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    def take(split, n):
        ds = load_dataset(DATASET, split=split, streaming=True)
        rows = []
        for i, r in enumerate(ds):
            if i >= n:
                break
            rows.append((r.get("hard_text") or r.get("text") or "",
                         int(r["profession"]), int(r["gender"])))
        return rows

    # Oversample then filter, because restricting to the top-N professions drops
    # rows; the repo's task construction takes the most frequent professions by
    # training count, so the same rule is applied here.
    tr = take("train", n_train * 3)
    te = take("test", n_test * 3)
    top = {p for p, _ in collections.Counter(p for _, p, _ in tr).most_common(N_CLASSES)}
    tr = [x for x in tr if x[1] in top][:n_train]
    te = [x for x in te if x[1] in top][:n_test]
    lab = {p: i for i, p in enumerate(sorted(top))}

    ytr = np.array([lab[p] for _, p, _ in tr])
    gtr = np.array([g for _, _, g in tr])
    yte = np.array([lab[p] for _, p, _ in te])
    gte = np.array([g for _, _, g in te])

    tok = BertTokenizerFast.from_pretrained(ENCODER)
    bert = BertModel.from_pretrained(ENCODER).eval()

    def embed(texts, length):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                batch = tok(texts[i:i + 64], max_length=length,
                            padding="max_length", truncation=True,
                            add_special_tokens=True, return_tensors="pt")
                out.append(bert(**batch).pooler_output.numpy())
        return np.vstack(out)

    rows = []
    for length in budgets:
        etr = embed([t for t, _, _ in tr], length)
        ete = embed([t for t, _, _ in te], length)
        prof = LogisticRegression(max_iter=3000).fit(etr, ytr)
        gend = LogisticRegression(max_iter=3000).fit(etr, gtr)
        rows.append({
            "encoder": ENCODER,
            "probe": "linear_on_frozen_pooled",
            "n_train": len(tr),
            "n_test": len(te),
            "n_classes": len(top),
            "max_length": length,
            "configured": int(length == 32),
            "profession_balacc": f"{balanced_accuracy_score(yte, prof.predict(ete)):.4f}",
            "gender_auc": f"{roc_auc_score(gte, gend.predict_proba(ete)[:, 1]):.4f}",
        })
        print(f"  max_length={length:3d}  profession BalAcc "
              f"{rows[-1]['profession_balacc']}  gender AUC {rows[-1]['gender_auc']}",
              flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=int, nargs="*", default=list(BUDGETS))
    args = ap.parse_args()
    try:
        rows = run(tuple(args.budgets))
    except Exception as exc:
        print(f"probe unavailable ({type(exc).__name__}: {exc})")
        return 1
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
