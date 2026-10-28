"""
retune_head.py — Post-hoc head recalibration on frozen encoder features.

Fits a fresh logistic regression head on frozen encoder features from memory,
separating encoder forgetting from head recency bias.

Two head variants (same encoder features, different reweighting):
  vanilla — inverse class frequency (ERM head; class_weight='balanced')
  fair    — inverse (class, group) cell frequency (Kamiran-Calders reweighing)

Usage (run from bias_in_bios/results):
    python ../retune_head.py --seed 0                    # retained-data recovery
    python ../retune_head.py --seed 0 --shared-reference # shared validation data

The shared-reference protocol fits every method's head on the same held-out
validation examples, removing replay-buffer composition from cross-method
encoder comparisons. Its outputs use the ``results_probe_shared*`` prefix.
"""

import os
import sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from model import _x_tensor
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
from train import OUT, METHODS, N_TASKS, CHECKPOINT_PATH, DATA_SEED

# Use the GPU when available — BERT feature extraction over the full
# test sets is far too slow on CPU. Falls back to CPU automatically.
DEVICE_EVAL = DEVICE

# Encoder forwards at seq-32 are launch-bound, not compute-bound: on a large
# card a 256-row batch leaves most of the GPU idle. Eval-mode outputs do not
# depend on batch size -- BERT normalises with LayerNorm, and the head's
# BatchNorm uses running statistics here -- so this is a pure throughput knob.
EVAL_BATCH = int(os.environ.get("CLFAIR_EVAL_BATCH", "256"))


def _extract_features(model, X_tokens, device="cpu", batch_size=None):
    """Extract BERT encoder features (before the classification head)."""
    batch_size = batch_size or EVAL_BATCH
    model.eval()
    model.to(device)
    X_t = _x_tensor(X_tokens)
    feats = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            batch = X_t[i:i + batch_size].to(device)
            feats.append(model.encode(batch).cpu())
    return torch.cat(feats).numpy()


def _extract_features_mlp(model, X, device="cpu", batch_size=None):
    """Extract penultimate-layer features from an MLP (before final linear)."""
    batch_size = batch_size or EVAL_BATCH
    model.eval()
    model.to(device)
    X_t = _x_tensor(X)
    feats = []
    # Methods that wrap an encoder instead of exposing a plain Sequential
    # (FaIRL) have no .net. _FAIRLNet provides encode(), which also applies
    # the L2 normalisation its classifier was trained against, so it is the
    # correct penultimate representation -- not self.encoder alone.
    penultimate = model.encode if hasattr(model, "encode") else model.net[:-1]
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            batch = X_t[i:i + batch_size].to(device)
            feats.append(penultimate(batch).cpu())
    return torch.cat(feats).numpy()


def extract_features(model, X, device="cpu", batch_size=None, bert_mode=False):
    if bert_mode:
        return _extract_features(model, X, device, batch_size)
    return _extract_features_mlp(model, X, device, batch_size)


def _inv_freq_weights(*cols: np.ndarray) -> np.ndarray:
    """Per-sample weights = 1/cell_frequency, mean-normalised.

    One column (y):    inverse class frequency — identical to class_weight='balanced'.
    Two columns (y,s): inverse (class, group) cell frequency — Kamiran-Calders reweighing.
    """
    keys = np.stack(cols, axis=1)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv.ravel()]
    return w * (len(w) / w.sum())


PROBE_JOBS = int(os.environ.get("CLFAIR_PROBE_JOBS", "1"))
METRIC_KEYS = ("acc", "bal_acc", "eo", "dp", "tpr", "acc_parity", "worst_group")


def _fit_and_score(feats_mem, y_mem, s_mem, test_feats, y_tests, s_tests):
    """Fit both heads on one method's features and score every eval task.

    The entire CPU half of the probe: two logistic regressions and their
    metrics. Sequential and parallel execution both call this, so the numbers
    cannot depend on which scheduling path ran -- only on the inputs.
    """
    head_weights = {
        "vanilla": _inv_freq_weights(y_mem),
        "fair":    _inv_freq_weights(y_mem, s_mem),
    }
    out = {}
    for variant, weights in head_weights.items():
        head = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        head.fit(feats_mem, y_mem, sample_weight=weights)
        rows = {k: [] for k in METRIC_KEYS}
        for feats, y_test, s_test in zip(test_feats, y_tests, s_tests):
            preds = head.predict(feats)
            rows["acc"].append(task_accuracy(y_test, preds))
            rows["bal_acc"].append(balanced_accuracy_score(y_test, preds))
            rows["eo"].append(equalized_odds_gap(y_test, preds, s_test))
            rows["dp"].append(demographic_parity_gap(preds, s_test))
            rows["tpr"].append(tpr_gap_rms(y_test, preds, s_test))
            rows["acc_parity"].append(accuracy_parity_gap(y_test, preds, s_test))
            rows["worst_group"].append(worst_group_balanced_accuracy(y_test, preds, s_test))
        out[variant] = rows
    return out


def run():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"No checkpoint at {CHECKPOINT_PATH}. Run train.py first.")
        return

    print("Loading data and checkpoint...")
    # See probe_leakage: must rebuild the split with the run's data seed.
    tasks, scaler = factory.get_tasks(seed=DATA_SEED)
    _bert_mode = factory.is_bert_task(tasks[0])
    input_dim = tasks[0]["X_train"].shape[1] if not _bert_mode else None

    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False, map_location="cpu")
    n_done = ckpt['last_completed_task'] + 1
    print(f"Checkpoint has {n_done}/{N_TASKS} tasks complete.")

    # Construct all method objects and restore state
    joint_retrain = JointRetrain(input_dim, device=DEVICE_EVAL, bert_mode=_bert_mode,
                                 bert_lr=config.JOINT_BERT_LR)
    finetune      = Finetune(input_dim, device=DEVICE_EVAL, bert_mode=_bert_mode,
                             bert_lr=config.FINETUNE_BERT_LR)
    agem          = AGEM(input_dim, device=DEVICE_EVAL, bert_mode=_bert_mode,
                         bert_lr=config.AGEM_BERT_LR)
    derpp         = DERpp(input_dim, device=DEVICE_EVAL, bert_mode=_bert_mode, bert_lr=config.DERPP_BERT_LR)
    group_dro     = GroupDRO(input_dim, device=DEVICE_EVAL, bert_mode=_bert_mode, bert_lr=config.GROUPDRO_BERT_LR)
    fairl         = FaIRL(input_dim, device=DEVICE_EVAL, bert_mode=_bert_mode, bert_lr=config.FAIRL_BERT_LR)
    fsw           = FSW(input_dim, device=DEVICE_EVAL, bert_mode=_bert_mode, bert_lr=config.FSW_BERT_LR)
    fair_gem      = FAIRGEM(input_dim, device=DEVICE_EVAL, use_topk_dro=False, use_disparity=True,
                            dynamic_eps=True, bert_mode=_bert_mode, bert_lr=config.FAIRGEM_BERT_LR)
    fair_derpp    = FairDERpp(input_dim, device=DEVICE_EVAL, bert_mode=_bert_mode, bert_lr=config.FAIRDERPP_BERT_LR)

    from train import _restore_state
    _restore_state(ckpt['state'], joint_retrain, finetune, agem, derpp,
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

    shared_reference = "--shared-reference" in sys.argv

    # Gather memory data for head retraining.
    # Methods with episodic memory: use their stored samples.
    # Methods without memory (joint_retrain, finetune, group_dro): use training
    # data from all tasks as the "memory" — they had access to it during training.
    print("Gathering memory/training data for head retraining...")
    X_key = "token_ids_train" if _bert_mode else "X_train"
    X_test_key = "token_ids_test" if _bert_mode else "X_test"

    # Collect all training data (for methods without memory)
    all_X_train = np.concatenate([t[X_key] for t in tasks[:n_done]])
    all_y_train = np.concatenate([t["y_train"] for t in tasks[:n_done]])
    all_s_train = np.concatenate([t["s_train"] for t in tasks[:n_done]])
    all_X_val = np.concatenate([t["token_ids_val" if _bert_mode else "X_val"]
                                for t in tasks[:n_done]])
    all_y_val = np.concatenate([t["y_val"] for t in tasks[:n_done]])
    all_s_val = np.concatenate([t["s_val"] for t in tasks[:n_done]])

    def _get_memory_data(method_name, method_obj):
        """Get (X, y, s) arrays for head retraining."""
        if shared_reference:
            return all_X_val, all_y_val, all_s_val
        if hasattr(method_obj, 'memory') and method_obj.memory.total_size() > 0:
            mem = method_obj.memory
            X_m = torch.cat(mem.X_buf).cpu().numpy()
            y_m = torch.cat(mem.y_buf).cpu().numpy()
            s_m = torch.cat(mem.s_buf).cpu().numpy()
            return X_m, y_m, s_m
        # No memory — subsample training data to ~same size as memory methods
        n_mem = config.MEMORY_SIZE_PER_TASK * n_done
        if len(all_X_train) > n_mem:
            # Seeded so the memory-less methods' probe numbers are reproducible.
            idx = np.random.RandomState(DATA_SEED).choice(len(all_X_train), n_mem, replace=False)
            return all_X_train[idx], all_y_train[idx], all_s_train[idx]
        return all_X_train, all_y_train, all_s_train

    protocol = "shared validation reference" if shared_reference else "retained data"
    print(f"Probe protocol: {protocol}")

    # Two head variants, each a full set of metric matrices. They share the same
    # frozen encoder features and differ ONLY in the head's training reweighting:
    #   vanilla — inverse class frequency (ERM head).
    #   fair    — inverse (class, group) cell frequency (Kamiran-Calders reweighing);
    #             the fairness-aware head. Together with the as-trained metrics in
    #             results_test.npz these form the ablation triple:
    #             as-trained → vanilla retune → fair retune.
    VARIANTS = ("vanilla", "fair")
    mats = {v: {k: {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS}
                for k in METRIC_KEYS}
            for v in VARIANTS}

    t_idx = n_done - 1

    # GPU half: extract every method's features first, sequentially. Nine
    # concurrent BERT passes would multiply activation memory for no gain --
    # the encoder work is already fast (a few minutes); the fits are not.
    payloads = {}
    for m_name, m_obj in method_objs.items():
        print(f"\n  Extracting {m_name}...")
        model = m_obj.model
        X_mem, y_mem, s_mem = _get_memory_data(m_name, m_obj)
        print(f"    Memory: {len(X_mem)} samples, {len(np.unique(y_mem))} classes")
        feats_mem  = extract_features(model, X_mem, device=DEVICE_EVAL, bert_mode=_bert_mode)
        test_feats = [extract_features(model, tasks[e][X_test_key], device=DEVICE_EVAL,
                                       bert_mode=_bert_mode) for e in range(n_done)]
        payloads[m_name] = (feats_mem, y_mem, s_mem, test_feats)

    y_tests = [tasks[e]["y_test"] for e in range(n_done)]
    s_tests = [tasks[e]["s_test"] for e in range(n_done)]

    # CPU half: 18 lbfgs fits with no shared state. CLFAIR_PROBE_JOBS>1 spreads
    # them across cores; 1 (the default) runs them sequentially.
    names = list(payloads)
    args = [(payloads[m][0], payloads[m][1], payloads[m][2], payloads[m][3],
             y_tests, s_tests) for m in names]
    if PROBE_JOBS == 1:
        results = [_fit_and_score(*a) for a in args]
    else:
        from joblib import Parallel, delayed
        print(f"\n  Fitting {2 * len(names)} heads across {PROBE_JOBS} workers...")
        results = Parallel(n_jobs=PROBE_JOBS, backend="loky")(
            delayed(_fit_and_score)(*a) for a in args)

    for m_name, res in zip(names, results):
        for variant, rows in res.items():
            M = mats[variant]
            for key in METRIC_KEYS:
                for e, value in enumerate(rows[key]):
                    M[key][m_name][t_idx, e] = value
        row = [f"{mats['vanilla']['bal_acc'][m_name][t_idx, e]:.3f}" for e in range(n_done)]
        print(f"    [{m_name:15s}] vanilla bal_acc {row}")

    # Save each variant in the schema plot_results.py expects.
    task_names = [t.get("state", t.get("attr", f"task_{i}")) for i, t in enumerate(tasks)]
    if shared_reference:
        out_paths = {
            "vanilla": os.path.join(OUT, "results_probe_shared.npz"),
            "fair": os.path.join(OUT, "results_probe_shared_fair.npz"),
        }
    else:
        out_paths = {
            "vanilla": os.path.join(OUT, "results_retune.npz"),
            "fair": os.path.join(OUT, "results_retune_fair.npz"),
        }
    for variant, path in out_paths.items():
        M = mats[variant]
        np.savez(path, acc=M["acc"], bal_acc=M["bal_acc"], eo=M["eo"], dp=M["dp"],
                 tpr=M["tpr"], acc_parity=M["acc_parity"], worst_group=M["worst_group"],
                 methods=METHODS, states=task_names)
        print(f"\nResults saved to {path}")

    print(f"\n{'-'*90}")
    print(f"{'Method':16s}  {'BalAcc':>14}  {'WrstGrp':>14}  {'TPRgap':>14}  {'EOgap':>14}")
    print(f"{'':16s}  {'vanilla -> fair':>14}  {'vanilla -> fair':>14}  {'vanilla -> fair':>14}  {'vanilla -> fair':>14}")
    print(f"{'-'*90}")
    for m in METHODS:
        def mean(variant, key):
            return np.nanmean(mats[variant][key][m][t_idx, :n_done])
        print(f"  {m:14s}  {mean('vanilla','bal_acc'):.3f} -> {mean('fair','bal_acc'):.3f}  "
              f"  {mean('vanilla','worst_group'):.3f} -> {mean('fair','worst_group'):.3f}  "
              f"  {mean('vanilla','tpr'):.3f} -> {mean('fair','tpr'):.3f}  "
              f"  {mean('vanilla','eo'):.3f} -> {mean('fair','eo'):.3f}")


if __name__ == "__main__":
    run()
