"""
eval_readouts.py — kNN readout on the shared reference set.

Adds a non-parametric readout to the three produced elsewhere:

    memory-refit   logistic head, each method's own retained buffer  (retune_head.py)
    shared-refit   logistic head, shared held-out validation data    (retune_head.py --shared-reference)
    NCM            class centroids, all training data seen so far    (eval_ncm.py)

Memory-refit and NCM differ in readout type *and* in fitting set, so a ranking
disagreement between them cannot be attributed to the readout alone. Holding the
data fixed at the shared reference and varying only the readout is what isolates
that axis; this script supplies the non-parametric point on it.

The reference set is the same `all_X_val` that retune_head.py --shared-reference
fits on, so the two are directly comparable. kNN takes no sample weights, so the
reference set is class-balanced by seeded subsampling instead — the analogue of
the inverse-class-frequency weighting the logistic heads use.

Usage (run from bias_in_bios/results):
    python ../eval_readouts.py [--seed N] [--k 10]

Output: results_knn_shared.npz  (same schema as results_probe_shared.npz)
"""

import os
import sys

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
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
from train import OUT, METHODS, N_TASKS, CHECKPOINT_PATH, DATA_SEED, _restore_state
from retune_head import extract_features

DEFAULT_K = 10
METRIC_KEYS = ("acc", "bal_acc", "eo", "dp", "tpr", "acc_parity", "worst_group")


def _arg_int(flag: str, default: int) -> int:
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 >= len(sys.argv):
            raise SystemExit(f"{flag} requires an integer argument")
        return int(sys.argv[i + 1])
    return default


def _class_balanced_subsample(y: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Indices of an equal-count-per-class subsample, at the rarest class's count.

    kNN has no sample-weight equivalent of the logistic heads' inverse-frequency
    reweighting, so balance is imposed on the reference set itself. Capped by the
    rarest seen class, which is what makes every class contribute equally to the
    neighbourhood vote.
    """
    classes, counts = np.unique(y, return_counts=True)
    n_per = int(counts.min())
    keep = []
    for c in classes:
        idx = np.flatnonzero(y == c)
        keep.append(rng.choice(idx, n_per, replace=False) if len(idx) > n_per else idx)
    out = np.concatenate(keep)
    rng.shuffle(out)
    return out


def main():
    k = _arg_int("--k", DEFAULT_K)

    print("kNN readout on the shared reference set")
    print("=" * 80)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"No checkpoint at {CHECKPOINT_PATH}. Run train.py first.")
        return

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

    X_val_key  = "token_ids_val"  if _bert_mode else "X_val"
    X_test_key = "token_ids_test" if _bert_mode else "X_test"

    # The identical reference set retune_head.py --shared-reference fits on.
    all_X_val = np.concatenate([t[X_val_key] for t in tasks[:n_done]])
    all_y_val = np.concatenate([t["y_val"] for t in tasks[:n_done]])

    rng = np.random.RandomState(DATA_SEED)
    keep = _class_balanced_subsample(all_y_val, rng)
    ref_X, ref_y = all_X_val[keep], all_y_val[keep]
    print(f"Reference set: {len(ref_X)} samples, {len(np.unique(ref_y))} classes "
          f"(class-balanced from {len(all_X_val)}), k={k}\n")

    mats = {key: {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS}
            for key in METRIC_KEYS}
    t_idx = n_done - 1

    for m_name, m_obj in method_objs.items():
        print(f"  {m_name}...")
        model = m_obj.model
        ref_feats  = extract_features(model, ref_X, device=DEVICE, bert_mode=_bert_mode)
        test_feats = [extract_features(model, tasks[e][X_test_key], device=DEVICE,
                                       bert_mode=_bert_mode) for e in range(n_done)]

        knn = KNeighborsClassifier(n_neighbors=min(k, len(ref_feats)))
        knn.fit(ref_feats, ref_y)

        for e in range(n_done):
            y_test, s_test = tasks[e]["y_test"], tasks[e]["s_test"]
            preds = knn.predict(test_feats[e])
            mats["acc"][m_name][t_idx, e]         = task_accuracy(y_test, preds)
            mats["bal_acc"][m_name][t_idx, e]     = balanced_accuracy_score(y_test, preds)
            mats["eo"][m_name][t_idx, e]          = equalized_odds_gap(y_test, preds, s_test)
            mats["dp"][m_name][t_idx, e]          = demographic_parity_gap(preds, s_test)
            mats["tpr"][m_name][t_idx, e]         = tpr_gap_rms(y_test, preds, s_test)
            mats["acc_parity"][m_name][t_idx, e]  = accuracy_parity_gap(y_test, preds, s_test)
            mats["worst_group"][m_name][t_idx, e] = worst_group_balanced_accuracy(y_test, preds, s_test)

        row = [f"{mats['bal_acc'][m_name][t_idx, e]:.3f}" for e in range(n_done)]
        print(f"    bal_acc {row}  mean={np.nanmean(mats['bal_acc'][m_name][t_idx, :n_done]):.3f}")

    task_names = [t.get("state", t.get("attr", f"task_{i}")) for i, t in enumerate(tasks)]
    path = os.path.join(OUT, "results_knn_shared.npz")
    np.savez(path, methods=METHODS, states=task_names,
             **{key: mats[key] for key in METRIC_KEYS})
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
