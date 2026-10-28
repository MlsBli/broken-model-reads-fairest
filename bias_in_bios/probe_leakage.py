"""
probe_leakage.py — Gender leakage probe for trained CL models.

Trains a logistic regression to predict gender from each method's penultimate
representations. Higher accuracy = more gender information encoded = more leakage.
A fair representation gives ~50% accuracy (chance).

Usage (run from bias_in_bios/results):
    python ../probe_leakage.py --seed 0          # val set
    python ../probe_leakage.py --seed 0 --test   # test set
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))

import factory
import config
from config import DEVICE, SEED
from model import BertMLP, MLP, _x_tensor
from baselines_neural import JointRetrain, Finetune, AGEM, DERpp
from baselines_fairness import GroupDRO
from baselines_fairl import FaIRL
from fsw import FSW
from fairgem import FAIRGEM
from fairderpp import FairDERpp
from train import OUT, METHODS, CHECKPOINT_PATH, N_TASKS, DATA_SEED, _restore_state


def _make_probe():
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        tol=1e-3,
        random_state=SEED,
    )


def _extract_penultimate(model: nn.Module, X: np.ndarray,
                         device: str = "cpu", batch_size: int = 256) -> np.ndarray:
    """Extract penultimate-layer activations (before the final Linear)."""
    model.eval()
    X_t = _x_tensor(X)

    if isinstance(model, BertMLP):
        # BertMLP: encode → head[:-1] (everything except final Linear)
        penult = nn.Sequential(*list(model.head.children())[:-1])
        feats = []
        with torch.no_grad():
            for i in range(0, len(X_t), batch_size):
                batch = X_t[i:i + batch_size].to(device)
                enc = model.encode(batch)
                feats.append(penult(enc).cpu().numpy())
    elif isinstance(model, MLP):
        # MLP: net[:-1] (everything except final Linear)
        penult = nn.Sequential(*list(model.net.children())[:-1])
        feats = []
        with torch.no_grad():
            for i in range(0, len(X_t), batch_size):
                batch = X_t[i:i + batch_size].to(device)
                feats.append(penult(batch).cpu().numpy())
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

    return np.concatenate(feats, axis=0)


def _extract_fairl_encoder(fairl_method, X: np.ndarray,
                           device: str = "cpu", batch_size: int = 256) -> np.ndarray:
    """Extract FaIRL encoder representations (before classifier head)."""
    model = fairl_method.model
    model.eval()
    X_t = _x_tensor(X)

    feats = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            batch = X_t[i:i + batch_size].to(device)
            # _FAIRLNet.encode() returns L2-normalised encoder output —
            # this IS the penultimate representation (before the classifier head).
            feats.append(model.encode(batch).cpu().numpy())

    return np.concatenate(feats, axis=0)


def _probe_accuracy(X_train, s_train, X_test, s_test):
    """Train probe and return (accuracy, AUC)."""
    probe = _make_probe()
    probe.fit(X_train, s_train)
    preds = probe.predict(X_test)
    probs = probe.predict_proba(X_test)[:, 1]
    acc = accuracy_score(s_test, preds)
    auc = roc_auc_score(s_test, probs)
    return acc, auc


def main():
    use_test = "--test" in sys.argv
    split = "test" if use_test else "val"

    print(f"Gender Leakage Probe (eval on {split} set)")
    print("=" * 80)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"No checkpoint found at {CHECKPOINT_PATH}. Run train.py first.")
        return

    print("Loading data...")
    # The run's data seed, not config.SEED: rebuilding the split with a
    # different seed would put encoder-training rows into the probe's eval split.
    tasks, _ = factory.get_tasks(need_frozen_embeds=True, seed=DATA_SEED)
    _bert_mode = factory.is_bert_task(tasks[0])
    input_dim = tasks[0]["X_train"].shape[1]

    print("Loading checkpoint...")
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False)
    n_done = ckpt["last_completed_task"] + 1
    print(f"  {n_done}/{N_TASKS} tasks complete in checkpoint.\n")

    # Construct all methods and restore state
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

    _restore_state(
        ckpt["state"], joint_retrain, finetune, agem, derpp,
        group_dro, fairl, fsw, fair_gem, fair_derpp,
    )

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

    # 1. Baseline: frozen BERT embeddings
    print("-" * 80)
    print("Frozen BERT embeddings (shared input - baseline leakage)")
    print("-" * 80)

    X_tr_all = np.concatenate([t["X_train"] for t in tasks[:n_done]])
    s_tr_all = np.concatenate([t["s_train"] for t in tasks[:n_done]])

    header = f"  {'Task':30s}  {'Acc':>6}  {'AUC':>6}  {'N_train':>7}  {'N_eval':>6}  {'%F':>5}"
    print(header)

    baseline_results = {}
    for e in range(n_done):
        t = tasks[e]
        X_ev = t[f"X_{split}"]
        s_ev = t[f"s_{split}"]
        acc, auc = _probe_accuracy(X_tr_all, s_tr_all, X_ev, s_ev)
        pct_f = 100.0 * s_ev.mean()
        name = t.get("state", f"task_{e}")[:30]
        print(f"  {name:30s}  {acc:.4f}  {auc:.4f}  {len(s_tr_all):>7}  {len(s_ev):>6}  {pct_f:>4.1f}")
        baseline_results[e] = {"acc": acc, "auc": auc}

    avg_acc = np.mean([v["acc"] for v in baseline_results.values()])
    avg_auc = np.mean([v["auc"] for v in baseline_results.values()])
    print(f"  {'AVERAGE':30s}  {avg_acc:.4f}  {avg_auc:.4f}")

    # 2. Per-method penultimate representations
    results = {}  # results[method][task_idx] = {acc, auc}

    for mname in METHODS:
        method = method_objs[mname]
        if method is None:
            continue

        print(f"\n{'-' * 80}")
        print(f"{mname} - penultimate layer representations")
        print(f"{'-' * 80}")
        print(header)

        results[mname] = {}

        # Build training features from all seen tasks
        if _bert_mode:
            X_key = "token_ids_train"
        else:
            X_key = "X_train"

        # Accumulate training data for the probe across all completed tasks
        feat_tr_list, s_tr_list = [], []
        for t_idx in range(n_done):
            t = tasks[t_idx]
            X_raw = t[X_key]
            if mname == "fairl":
                feat = _extract_fairl_encoder(method, X_raw, DEVICE)
            else:
                feat = _extract_penultimate(method.model, X_raw, DEVICE)
            feat_tr_list.append(feat)
            s_tr_list.append(t["s_train"])

        feat_tr = np.concatenate(feat_tr_list)
        s_tr = np.concatenate(s_tr_list)

        for e in range(n_done):
            t = tasks[e]
            X_ev_key = f"token_ids_{split}" if _bert_mode else f"X_{split}"
            X_ev_raw = t[X_ev_key]
            s_ev = t[f"s_{split}"]

            if mname == "fairl":
                feat_ev = _extract_fairl_encoder(method, X_ev_raw, DEVICE)
            else:
                feat_ev = _extract_penultimate(method.model, X_ev_raw, DEVICE)

            acc, auc = _probe_accuracy(feat_tr, s_tr, feat_ev, s_ev)
            pct_f = 100.0 * s_ev.mean()
            name = t.get("state", f"task_{e}")[:30]
            print(f"  {name:30s}  {acc:.4f}  {auc:.4f}  {len(s_tr):>7}  {len(s_ev):>6}  {pct_f:>4.1f}")
            results[mname][e] = {"acc": acc, "auc": auc}

        avg_acc = np.mean([v["acc"] for v in results[mname].values()])
        avg_auc = np.mean([v["auc"] for v in results[mname].values()])
        print(f"  {'AVERAGE':30s}  {avg_acc:.4f}  {avg_auc:.4f}")

    # Summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY - Average gender probe accuracy (chance = 50%)")
    print(f"{'=' * 80}")
    print(f"  {'Method':20s}  {'Probe Acc':>10}  {'Probe AUC':>10}  {'delta  vs Input':>10}")
    print(f"  {'-' * 55}")

    base_avg = np.mean([v["acc"] for v in baseline_results.values()])
    print(f"  {'frozen_bert_input':20s}  {base_avg:>10.4f}  "
          f"{np.mean([v['auc'] for v in baseline_results.values()]):>10.4f}  {'-':>10}")

    for mname in METHODS:
        if mname not in results:
            continue
        m_avg_acc = np.mean([v["acc"] for v in results[mname].values()])
        m_avg_auc = np.mean([v["auc"] for v in results[mname].values()])
        delta = m_avg_acc - base_avg
        print(f"  {mname:20s}  {m_avg_acc:>10.4f}  {m_avg_auc:>10.4f}  {delta:>+10.4f}")

    # Save results
    out = {
        "baseline": baseline_results,
        "methods": results,
        "split": split,
        "n_tasks": n_done,
    }
    np.save(os.path.join(OUT, "results_leakage.npy"), out, allow_pickle=True)
    print(f"\nResults saved to results_leakage.npy")


if __name__ == "__main__":
    main()
