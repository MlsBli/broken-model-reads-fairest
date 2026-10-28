"""
eval_ncm.py — Nearest Class Mean evaluation on trained CL models.

Bypasses the linear head; assigns each test sample to its nearest class centroid.
Useful for diagnosing head recency bias vs. encoder forgetting.

Usage (run from bias_in_bios/results):
    python ../eval_ncm.py --seed 0       # test set (default)
    python ../eval_ncm.py --seed 0 --val # val set

Output: results_ncm.npy, results_ncm_full.npz
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

import factory
import config
from config import DEVICE, N_CLASSES
from model import MLP, _x_tensor
from baselines_neural import JointRetrain, Finetune, AGEM, DERpp
from baselines_fairness import GroupDRO
from baselines_fairl import FaIRL
from fsw import FSW
from fairgem import FAIRGEM
from fairderpp import FairDERpp
from train import OUT, METHODS, CHECKPOINT_PATH, N_TASKS, DATA_SEED, _restore_state
from sklearn.metrics import balanced_accuracy_score
from metrics import (task_accuracy, accuracy_parity_gap, demographic_parity_gap,
                     equalized_odds_gap, tpr_gap_rms,
                     worst_group_balanced_accuracy)


def _remap(d):
    return d


# Feature extraction
def _extract_features(model, X, device, batch_size=None):
    """Extract encoder features, bypassing the classification head."""
    batch_size = batch_size or int(os.environ.get("CLFAIR_EVAL_BATCH", "256"))
    model.eval()
    X_t = _x_tensor(X)
    penult = None
    if isinstance(model, MLP):
        penult = nn.Sequential(*list(model.net.children())[:-1])

    feats = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            batch = X_t[i:i + batch_size].to(device)
            if penult is not None:
                feats.append(penult(batch).cpu().numpy())
            elif hasattr(model, 'encode'):
                feats.append(model.encode(batch).cpu().numpy())
            else:
                raise ValueError(f"Cannot extract features from {type(model)}")
    return np.concatenate(feats)


# NCM
def _compute_centroids(model, tasks, n_done, device, bert_mode):
    """Compute per-class mean feature vectors from all training data seen so far."""
    X_key = "token_ids_train" if bert_mode else "X_train"
    class_feats = defaultdict(list)
    for t_idx in range(n_done):
        task  = tasks[t_idx]
        X     = task[X_key]
        y     = task["y_train"]
        feats = _extract_features(model, X, device)
        for c in np.unique(y):
            class_feats[int(c)].append(feats[y == c])
    return {c: np.concatenate(f).mean(axis=0) for c, f in class_feats.items()}


def _ncm_predict(feats_test, centroids, seen_classes):
    """Assign each test sample to its nearest class centroid (L2 distance)."""
    classes = [c for c in seen_classes if c in centroids]
    if not classes:
        return np.zeros(len(feats_test), dtype=np.int64)
    centroid_mat = np.stack([centroids[c] for c in classes])   # [K, D]
    diffs = feats_test[:, None, :] - centroid_mat[None, :, :]  # [N, K, D]
    dists = np.linalg.norm(diffs, axis=-1)                      # [N, K]
    return np.array([classes[i] for i in dists.argmin(axis=1)])


# Main
def main():
    use_val = "--val" in sys.argv
    split   = "val" if use_val else "test"

    print(f"NCM Evaluation (eval on {split} set)")
    print("=" * 80)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"No checkpoint at {CHECKPOINT_PATH}. Run train.py first.")
        return

    print("Loading data...")
    # See probe_leakage: must rebuild the split with the run's data seed.
    # need_frozen_embeds is left to the factory, which derives it from
    # BIOS_BERT_FINETUNE: with a frozen encoder the model consumes the frozen
    # embeddings as X_train, so they must be computed; under fine-tuning it
    # resolves to False.
    tasks, _ = factory.get_tasks(seed=DATA_SEED)
    _bert_mode = factory.is_bert_task(tasks[0])
    input_dim  = tasks[0]["X_train"].shape[1]  # 768 in bert mode (ignored by BertMLP)

    print("Loading checkpoint...")
    ckpt   = torch.load(CHECKPOINT_PATH, weights_only=False)
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
        "joint_retrain": joint_retrain,
        "finetune":      finetune,
        "agem":          agem,
        "derpp":         derpp,
        "group_dro":     group_dro,
        "fairl":         fairl,
        "fsw":           fsw,
        "fair_gem":      fair_gem,
        "fair_derpp":    fair_derpp,
    }

    # Key for test/val inputs
    X_ev_key = ("token_ids_test" if not use_val else "token_ids_val") \
               if _bert_mode else f"X_{split}"

    # All configured classes seen by the final model.
    seen_classes = list(range(N_CLASSES))

    # Load linear-head results for side-by-side comparison.
    # Read from this run's output dir so the comparison uses the same seed.
    npz_path = os.path.join(OUT, "results_test.npz" if not use_val else "results.npz")
    try:
        lin_data = np.load(npz_path, allow_pickle=True)
        lin_bal  = _remap(lin_data["bal_acc"].item())
        has_linear = True
        print(f"Loaded linear-head balanced accuracy from {npz_path}\n")
    except FileNotFoundError:
        lin_bal    = {}
        has_linear = False
        print(f"No {npz_path} found - linear head comparison unavailable.\n")

    ncm_results = {}  # mname → np.array of per-task accuracy/F1
    # Full metric matrices for NCM evaluation (same format as results_test.npz)
    ncm_acc        = {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS}
    ncm_bal_acc    = {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS}
    ncm_eo         = {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS}
    ncm_dp         = {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS}
    ncm_tpr         = {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS}
    ncm_acc_parity  = {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS}
    ncm_worst_group = {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS}

    header = f"  {'Task':30s}  {'NCM':>8}  {'Linear':>8}  {'Δ':>8}  N"
    sep    = f"  {'─'*30}  {'─'*8}  {'─'*8}  {'─'*8}  ─"

    t_idx = n_done - 1

    for mname in METHODS:
        method = method_objs[mname]
        print(f"\n{'-' * 80}")
        print(f"{mname}")
        print(f"{'-' * 80}")
        print(f"  Building centroids from {n_done} tasks of training data...")

        model     = method.model
        centroids = _compute_centroids(model, tasks, n_done, DEVICE, _bert_mode)
        print(f"  {len(centroids)} class centroids built.")
        print(header)
        print(sep)

        ncm_vals = []
        lin_vals = []

        for e_idx in range(n_done):
            task     = tasks[e_idx]
            X_ev     = task[X_ev_key]
            y_ev     = task[f"y_{split}"]
            s_ev     = task[f"s_{split}"]
            name     = task.get("state", f"task_{e_idx}")[:30]

            feats   = _extract_features(model, X_ev, DEVICE)
            preds   = _ncm_predict(feats, centroids, seen_classes)
            ncm_val = float(balanced_accuracy_score(y_ev, preds))
            ncm_vals.append(ncm_val)

            # Store full metrics for NCM predictions
            ncm_acc[mname][t_idx, e_idx]     = task_accuracy(y_ev, preds)
            ncm_bal_acc[mname][t_idx, e_idx] = ncm_val
            ncm_eo[mname][t_idx, e_idx]      = equalized_odds_gap(y_ev, preds, s_ev)
            ncm_dp[mname][t_idx, e_idx]      = demographic_parity_gap(preds, s_ev)
            ncm_tpr[mname][t_idx, e_idx]     = tpr_gap_rms(y_ev, preds, s_ev)
            ncm_acc_parity[mname][t_idx, e_idx] = accuracy_parity_gap(y_ev, preds, s_ev)
            ncm_worst_group[mname][t_idx, e_idx] = worst_group_balanced_accuracy(y_ev, preds, s_ev)

            # Linear head balanced accuracy for comparison (same metric)
            if has_linear and mname in lin_bal and not np.isnan(lin_bal[mname][n_done - 1, e_idx]):
                lin_val = float(lin_bal[mname][n_done - 1, e_idx])
            else:
                lin_val = float("nan")
            lin_vals.append(lin_val)

            delta     = ncm_val - lin_val if not np.isnan(lin_val) else float("nan")
            d_str     = f"{delta:+.3f}" if not np.isnan(delta) else "    n/a"
            lin_str   = f"{lin_val:.4f}" if not np.isnan(lin_val) else "    n/a"
            print(f"  {name:30s}  {ncm_val:>8.4f}  {lin_str:>8}  {d_str:>8}  {len(y_ev)}")

        ncm_avg = float(np.mean(ncm_vals))
        lin_avg = float(np.nanmean(lin_vals)) if any(not np.isnan(v) for v in lin_vals) else float("nan")
        delta   = ncm_avg - lin_avg if not np.isnan(lin_avg) else float("nan")
        d_str   = f"{delta:+.3f}" if not np.isnan(delta) else "    n/a"
        lin_str = f"{lin_avg:.4f}" if not np.isnan(lin_avg) else "    n/a"
        print(sep)
        print(f"  {'AVERAGE':30s}  {ncm_avg:>8.4f}  {lin_str:>8}  {d_str:>8}")
        ncm_results[mname] = np.array(ncm_vals)

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY - NCM vs. Linear Head (final model evaluated on all tasks)")
    print(f"{'=' * 80}")
    print(f"  {'Method':20s}  {'NCM Avg':>9}  {'Lin Avg':>9}  {'delta ':>9}  Verdict")
    print(f"  {'-' * 70}")

    for mname in METHODS:
        if mname not in ncm_results:
            continue
        ncm_avg = float(np.mean(ncm_results[mname]))
        if has_linear and mname in lin_bal:
            lin_avg = float(np.nanmean([lin_bal[mname][n_done - 1, e] for e in range(n_done)]))
        else:
            lin_avg = float("nan")
        delta = ncm_avg - lin_avg if not np.isnan(lin_avg) else float("nan")

        if np.isnan(delta):
            verdict = "n/a"
        elif delta > 0.10:
            verdict = "Encoder intact — head forgot (NCM recovers well)"
        elif delta > 0.02:
            verdict = "Partial recovery — encoder mostly intact"
        elif delta > -0.02:
            verdict = "Similar — head and encoder in agreement"
        else:
            verdict = "Encoder also forgot"

        d_str   = f"{delta:+.4f}" if not np.isnan(delta) else "     n/a"
        lin_str = f"{lin_avg:.4f}" if not np.isnan(lin_avg) else "     n/a"
        print(f"  {mname:20s}  {ncm_avg:>9.4f}  {lin_str:>9}  {d_str:>9}  {verdict}")

    np.save(os.path.join(OUT, "results_ncm.npy"), ncm_results, allow_pickle=True)
    task_names = [t.get("state", t.get("attr", f"task_{i}")) for i, t in enumerate(tasks)]
    np.savez(os.path.join(OUT, "results_ncm_full.npz"),
             acc=ncm_acc, bal_acc=ncm_bal_acc, eo=ncm_eo, dp=ncm_dp, tpr=ncm_tpr,
             acc_parity=ncm_acc_parity, worst_group=ncm_worst_group,
             methods=METHODS, states=task_names)
    print(f"\nSaved to results_ncm.npy + results_ncm_full.npz")


if __name__ == "__main__":
    main()
