"""
eval_ttbudget.py — how much labelled data at test time repairs a collapsed model?

The deployed head collapses onto the most recent task; a head refit on the
replay buffer recovers most of the loss, but it spends data the method retained
during training. This script asks a test-time question instead: given a model
already deployed and already collapsed, how many labelled examples *from the
test stream* does repair cost?

Protocol, per method, at the final checkpoint:

  1. Each seen task's test set is split 50/50 into a repair pool and an eval
     pool, seeded by the run's data seed. Nothing fitted here is ever evaluated
     on, which is what separates this from the memory-refit probe.
  2. Budgets are nested draws from the union of the repair pools: the pool is
     shuffled once and the first B taken, so B=50 contains B=25. A larger budget
     therefore never has *different* data, only more of it, and the curve is not
     confounded with draw luck.
  3. The draw is uniform over the stream, not class-balanced, because that is
     what a deployment actually observes. The head is then fit with
     inverse-class-frequency sample weights -- the identical treatment
     retune_head.py gives its vanilla head, so the two are comparable.
  4. B=0 is the deployed head itself, scored on the same eval pool. At the final
     task every class has been seen, so logit masking is a no-op and applying
     the head to extracted features is exactly the deployed forward pass.

B = MEMORY_SIZE_PER_TASK * n_tasks is included deliberately: at that budget the
test-time repair set is the same size as the replay buffer the memory-refit head
uses, so the two protocols can be read against each other directly.

Usage (run from bias_in_bios/results):
    python ../eval_ttbudget.py [--seed N]
    python ../eval_ttbudget.py --aggregate --seeds 0 1 2 ...

Output: ttbudget.csv (per seed), ttbudget_summary.csv (aggregate)
"""

import csv
import glob
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from metrics import (task_accuracy, accuracy_parity_gap, demographic_parity_gap,
                     equalized_odds_gap, tpr_gap_rms, worst_group_balanced_accuracy)
from baselines_neural import JointRetrain, Finetune, AGEM, DERpp
from baselines_fairness import GroupDRO
from baselines_fairl import FaIRL
from fsw import FSW
from fairgem import FAIRGEM
from fairderpp import FairDERpp
import factory
import config
from config import DEVICE
from train import OUT, N_TASKS, CHECKPOINT_PATH, DATA_SEED, _restore_state
from retune_head import extract_features, _inv_freq_weights, EVAL_BATCH

METRIC_KEYS = ("acc", "bal_acc", "eo", "dp", "tpr", "acc_parity", "worst_group")
FIELDS = ["seed", "method", "budget", "n_classes_covered"] + list(METRIC_KEYS)


def _budget_grid(n_done: int) -> list[int]:
    """Nested budgets, ending at the replay-buffer size for direct comparison."""
    replay_equivalent = config.MEMORY_SIZE_PER_TASK * n_done
    grid = [0, 25, 50, 100, 250, 500, 1000, 2000, replay_equivalent]
    return sorted(set(b for b in grid if b >= 0))


def _deployed_preds(model, feats: np.ndarray, bert_mode: bool = False) -> np.ndarray:
    """Apply the model's own head to already-extracted encoder features.

    Equivalent to a full forward pass: extract_features returns exactly the
    tensor the head consumes, and at the final task seen_classes covers every
    class, so mask_logits is the identity.
    """
    # Three model shapes reach this point, and extract_features has already
    # produced exactly the tensor each one's head consumes:
    #   BertMLP    .head       fed by .encode()
    #   _FAIRLNet  .classifier fed by .encode(), which L2-normalises
    #   MLP        .net[-1]    fed by .net[:-1]
    if hasattr(model, "head"):
        head = model.head
    elif hasattr(model, "classifier"):
        head = model.classifier
    elif hasattr(model, "net"):
        head = model.net[-1]
    else:
        raise TypeError(
            f"{type(model).__name__} exposes no recognised classification head; "
            "add it here rather than letting the deployed-head anchor score "
            "against the wrong module.")
    model.eval()
    out = []
    with torch.no_grad():
        f = torch.tensor(feats, dtype=torch.float32)
        for i in range(0, len(f), EVAL_BATCH):
            out.append(head(f[i:i + EVAL_BATCH].to(DEVICE)).argmax(dim=1).cpu())
    preds = torch.cat(out).numpy()
    assert len(preds) == len(feats), (
        f"deployed-head anchor returned {len(preds)} predictions for "
        f"{len(feats)} rows -- batching bug, not a data problem")
    return preds


def _score(y, preds, s) -> dict:
    return {
        "acc":         task_accuracy(y, preds),
        "bal_acc":     balanced_accuracy_score(y, preds),
        "eo":          equalized_odds_gap(y, preds, s),
        "dp":          demographic_parity_gap(preds, s),
        "tpr":         tpr_gap_rms(y, preds, s),
        "acc_parity":  accuracy_parity_gap(y, preds, s),
        "worst_group": worst_group_balanced_accuracy(y, preds, s),
    }


def aggregate(seeds: list[int]) -> int:
    """Mean and sd across per-seed ttbudget.csv files."""
    rows = []
    for path in sorted(glob.glob(os.path.join("runs", "seed*", "ttbudget.csv"))):
        seed = int(os.path.basename(os.path.dirname(path)).replace("seed", ""))
        if seeds and seed not in seeds:
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            rows.extend(list(csv.DictReader(fh)))
    if not rows:
        print("No runs/seed*/ttbudget.csv found.")
        return 1

    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        grouped.setdefault((r["method"], r["budget"]), []).append(r)

    out_fields = ["method", "budget", "n_seeds"]
    for key in METRIC_KEYS:
        out_fields += [f"{key}_mean", f"{key}_std"]

    with open("ttbudget_summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=out_fields)
        writer.writeheader()
        for (method, budget) in sorted(grouped, key=lambda kv: (kv[0], int(kv[1]))):
            group = grouped[(method, budget)]
            row = {"method": method, "budget": budget, "n_seeds": len(group)}
            for key in METRIC_KEYS:
                vals = np.array([float(g[key]) for g in group], dtype=float)
                row[f"{key}_mean"] = f"{vals.mean():.6g}"
                row[f"{key}_std"]  = f"{vals.std(ddof=0):.6g}"
            writer.writerow(row)
    print(f"Wrote ttbudget_summary.csv ({len(grouped)} method x budget cells)")
    return 0


def main() -> int:
    if "--aggregate" in sys.argv:
        seeds = []
        if "--seeds" in sys.argv:
            for arg in sys.argv[sys.argv.index("--seeds") + 1:]:
                if arg.startswith("--"):
                    break
                seeds.append(int(arg))
        return aggregate(seeds)

    print("Test-time labelled-budget sweep")
    print("=" * 80)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"No checkpoint at {CHECKPOINT_PATH}. Run train.py first.")
        return 1

    print("Loading data...")
    # See probe_leakage: must rebuild the split with the run's data seed.
    tasks, _ = factory.get_tasks(seed=DATA_SEED)
    _bert_mode = factory.is_bert_task(tasks[0])
    input_dim = tasks[0]["X_train"].shape[1] if not _bert_mode else None

    print("Loading checkpoint...")
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False, map_location="cpu")
    n_done = ckpt["last_completed_task"] + 1
    print(f"  {n_done}/{N_TASKS} tasks complete.\n")

    joint_retrain = JointRetrain(input_dim, device=DEVICE, bert_mode=_bert_mode,
                                 bert_lr=config.JOINT_BERT_LR)
    finetune      = Finetune(input_dim, device=DEVICE, bert_mode=_bert_mode,
                             bert_lr=config.FINETUNE_BERT_LR)
    agem          = AGEM(input_dim, device=DEVICE, bert_mode=_bert_mode,
                         bert_lr=config.AGEM_BERT_LR)
    derpp         = DERpp(input_dim, device=DEVICE, bert_mode=_bert_mode,
                          bert_lr=config.DERPP_BERT_LR)
    group_dro     = GroupDRO(input_dim, device=DEVICE, bert_mode=_bert_mode,
                             bert_lr=config.GROUPDRO_BERT_LR)
    fairl         = FaIRL(input_dim, device=DEVICE, bert_mode=_bert_mode,
                          bert_lr=config.FAIRL_BERT_LR)
    fsw           = FSW(input_dim, device=DEVICE, bert_mode=_bert_mode,
                        bert_lr=config.FSW_BERT_LR)
    fair_gem      = FAIRGEM(input_dim, device=DEVICE, use_topk_dro=False,
                            use_disparity=True, dynamic_eps=True,
                            bert_mode=_bert_mode, bert_lr=config.FAIRGEM_BERT_LR)
    fair_derpp    = FairDERpp(input_dim, device=DEVICE, bert_mode=_bert_mode,
                              bert_lr=config.FAIRDERPP_BERT_LR)

    _restore_state(ckpt["state"], joint_retrain, finetune, agem, derpp,
                   group_dro, fairl, fsw, fair_gem, fair_derpp)

    method_objs = {
        "joint_retrain": joint_retrain, "finetune": finetune, "agem": agem,
        "derpp": derpp, "group_dro": group_dro, "fairl": fairl, "fsw": fsw,
        "fair_gem": fair_gem, "fair_derpp": fair_derpp,
    }

    X_test_key = "token_ids_test" if _bert_mode else "X_test"

    # 50/50 repair/eval split per task, seeded by the run's data seed. The eval
    # halves are never fitted on, at any budget.
    rng = np.random.RandomState(DATA_SEED)
    repair_idx, eval_idx = [], []
    for e in range(n_done):
        n = len(tasks[e]["y_test"])
        perm = rng.permutation(n)
        cut = n // 2
        repair_idx.append(perm[:cut])
        eval_idx.append(perm[cut:])

    repair_X = np.concatenate([tasks[e][X_test_key][repair_idx[e]] for e in range(n_done)])
    repair_y = np.concatenate([tasks[e]["y_test"][repair_idx[e]] for e in range(n_done)])
    repair_s = np.concatenate([tasks[e]["s_test"][repair_idx[e]] for e in range(n_done)])

    # One shuffle defines every budget: nested draws, so B=50 contains B=25.
    order = np.random.RandomState(DATA_SEED + 1).permutation(len(repair_X))
    repair_X, repair_y, repair_s = repair_X[order], repair_y[order], repair_s[order]

    budgets = _budget_grid(n_done)
    print(f"Repair pool: {len(repair_X)} samples.  Budgets: {budgets}\n")

    seed_label = str(DATA_SEED)
    out_rows = []

    for m_name, m_obj in method_objs.items():
        print(f"  {m_name}...")
        model = m_obj.model
        # Extracted once per method; every budget reuses these.
        repair_feats = extract_features(model, repair_X, device=DEVICE, bert_mode=_bert_mode)
        eval_feats = [extract_features(model, tasks[e][X_test_key][eval_idx[e]],
                                       device=DEVICE, bert_mode=_bert_mode)
                      for e in range(n_done)]
        eval_y = [tasks[e]["y_test"][eval_idx[e]] for e in range(n_done)]
        eval_s = [tasks[e]["s_test"][eval_idx[e]] for e in range(n_done)]

        for B in budgets:
            if B == 0:
                per_task = [_score(eval_y[e], _deployed_preds(model, eval_feats[e], _bert_mode),
                                   eval_s[e]) for e in range(n_done)]
                covered = len(np.unique(np.concatenate(eval_y)))
            else:
                fit_X, fit_y = repair_feats[:B], repair_y[:B]
                covered = len(np.unique(fit_y))
                if covered < 2:
                    print(f"    B={B}: only {covered} class in the draw - skipped")
                    continue
                head = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
                head.fit(fit_X, fit_y, sample_weight=_inv_freq_weights(fit_y))
                per_task = [_score(eval_y[e], head.predict(eval_feats[e]), eval_s[e])
                            for e in range(n_done)]

            row = {"seed": seed_label, "method": m_name, "budget": B,
                   "n_classes_covered": covered}
            for key in METRIC_KEYS:
                row[key] = f"{float(np.mean([p[key] for p in per_task])):.6g}"
            out_rows.append(row)
            print(f"    B={B:>5}  classes={covered:>2}  "
                  f"bal_acc={row['bal_acc']}  acc_parity={row['acc_parity']}")

    path = os.path.join(OUT, "ttbudget.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nResults saved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
