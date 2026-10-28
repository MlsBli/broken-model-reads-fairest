"""
tune.py — Optuna hyperparameter tuning for all neural baselines (BIOS).

Objective: bal_acc - EO_PENALTY * accuracy_parity_gap, averaged (or
recency-weighted) over N_TUNE_TASKS tasks with subsampled data.

Usage (run from bias_in_bios/results):
    python ../tune.py                              # tune all studies
    python ../tune.py --methods FaIRL,FAIR-GEM    # specific studies
    python ../tune.py --fresh                     # delete cached studies

Writes best values back to ./config.py in-place.
"""

import csv
import json
import os
import re
import numpy as np
import torch

from sklearn.metrics import recall_score

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from metrics import accuracy_parity_gap
from baselines_neural import JointRetrain, Finetune, DERpp, AGEM
from baselines_fairl import FaIRL
from baselines_fairness import GroupDRO
from fairgem import FAIRGEM
from fsw import FSW
from fairderpp import FairDERpp
from safety import validate_disparity_mode
import factory
from config import (DEVICE, SEED, DATASET, MLP_EPOCHS, BIOS_BERT_LR,
                    BIOS_BERT_EPOCHS, BIOS_STEP_SIZE, N_CLASSES,
                    MEMORY_SIZE_PER_TASK)

# Tuning scope
# All 5 tasks, so selection also sees the data-starved later tasks (the first
# 3 hold ~93% of the data), where the forgetting and fairness dynamics occur.
N_TUNE_TASKS        = 5
N_TUNE_SAMPLES      = 20_000   # max samples/task for non-BERT tuning runs
# 10k (~78 steps/task at batch 256 x 2 epochs): a subsample of task 0's
# ~700-step budget, but close to the real budget of tasks 1-4, which matters
# because bert_lr is sensitive to schedule length.
BERT_N_TUNE_SAMPLES = 10_000  # uniform selection budget across all 9 studies; recorded per-row in tuning_summary.csv
CONFIG_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
DB_PATH        = "tune.db"
SUMMARY_PATH   = "tuning_summary.csv"
EO_PENALTY     = 0.75

# Study-name suffix and manifest tag for this protocol (25 professions / 5
# tasks, tuning over all tasks at a 10k/task budget, run-seeded batch order).
# Studies stored under a different tag should not be resumed.
TUNE_VERSION   = "v6-bios-bert"

torch.manual_seed(SEED)
np.random.seed(SEED)


# Data
# Detect BERT mode once at import time so all objective functions agree.
# When BIOS_BERT_FINETUNE=True and the dataset is "bios", tasks will have
# token_ids_train (int token IDs [n,32]) rather than float embeddings (X_train).
_TASKS_SAMPLE = None  # populated lazily in run()
_BERT_MODE = False     # set in run() after tasks are loaded


def _disparity_mode() -> str:
    """Select the disparity surrogate (CLFAIR_DISPARITY_MODE) without changing the default config."""
    return validate_disparity_mode(
        os.environ.get("CLFAIR_DISPARITY_MODE", "unconditional")
    )


def _subsample(task, n):
    """Return (X, y, s) subsampled to at most n rows."""
    X_key = "token_ids_train" if _BERT_MODE else "X_train"
    X, y, s = task[X_key], task["y_train"], task["s_train"]
    if len(X) > n:
        idx = np.random.RandomState(SEED).choice(len(X), n, replace=False)
        return X[idx], y[idx], s[idx]
    return X, y, s


def _tune_sample_size() -> int:
    """Per-task sample cap: smaller for BERT to keep trial time reasonable."""
    return BERT_N_TUNE_SAMPLES if _BERT_MODE else N_TUNE_SAMPLES


def _tune_epochs() -> int:
    """Epochs per task during tuning — must match train.py's per-task epoch budget."""
    return BIOS_BERT_EPOCHS if _BERT_MODE else MLP_EPOCHS


def _patch_bert_lr(model, bert_lr: float):
    """Override the BERT param-group LR in an already-constructed optimizer."""
    if hasattr(model, "optimizer") and len(model.optimizer.param_groups) > 1:
        # Standard two-group AdamW from make_optimizer (BertMLP-based methods)
        model.optimizer.param_groups[0]["lr"] = bert_lr
    elif hasattr(model, "opt_enc") and hasattr(model, "encoder") and hasattr(model.encoder, "bert"):
        # FaIRL: BERT LR lives in the encoder optimizer
        model.opt_enc.param_groups[0]["lr"] = bert_lr


# Scoring helpers
def _seen_classes_after(n_tasks_seen: int) -> list[int] | None:
    if DATASET != "bios":
        return None
    hi = min(N_CLASSES, n_tasks_seen * BIOS_STEP_SIZE)
    return list(range(hi))


def _masked_predict(predict_fn, seen_classes: list[int] | None):
    if seen_classes is None:
        return predict_fn
    return lambda X: predict_fn(X, seen_classes=seen_classes)


def _task_score(task, predict_fn, x_key: str = None):
    """Per-task score: bal_acc − EO_PENALTY * accuracy_parity_gap."""
    if x_key is None:
        x_key = "token_ids_val" if _BERT_MODE else "X_val"
    Xv, yv, sv = task[x_key], task["y_val"], task["s_val"]
    preds = predict_fn(Xv)
    # Macro-recall restricted to the classes actually present in this task's
    # val labels. This equals balanced accuracy over present classes, but avoids
    # sklearn's "y_pred contains classes not in y_true" warning: in the
    # class-incremental setup the shared 25-way head legitimately predicts
    # professions from other tasks, which balanced_accuracy_score flags (it still
    # computes the same value, just noisily). Label-scoping matches metrics.py.
    labels = np.unique(yv)
    bal  = float(recall_score(yv, preds, labels=labels, average="macro", zero_division=0))
    fair = float(accuracy_parity_gap(yv, preds, sv))
    return bal - EO_PENALTY * fair


def _score(tasks_seen, predict_fn, x_key: str = None):
    """Uniform average of _task_score over all tasks seen so far."""
    return float(np.mean([_task_score(t, predict_fn, x_key) for t in tasks_seen]))


# Per-method objective functions
def _tune_joint_retrain(trial, tasks):
    bert_lr = trial.suggest_float("bert_lr", 5e-6, 5e-5, log=True)
    input_dim = tasks[0]["X_train"].shape[1]
    model = JointRetrain(
        input_dim, device=DEVICE, bert_mode=_BERT_MODE, bert_lr=bert_lr,
    )

    tasks_seen = []
    for task in tasks:
        X_tr, y_tr, _ = _subsample(task, _tune_sample_size())
        seen_now = _seen_classes_after(len(tasks_seen) + 1)
        model.train_task(X_tr, y_tr, epochs=_tune_epochs(), seen_classes=seen_now)
        tasks_seen.append(task)

    seen = _seen_classes_after(len(tasks_seen))
    score = _score(tasks_seen, _masked_predict(model.predict, seen))
    print(f"  [Joint Retrain] bert_lr={bert_lr:.1e} | score={score:.4f}")
    return score


def _tune_finetune(trial, tasks):
    bert_lr = trial.suggest_float("bert_lr", 5e-6, 5e-5, log=True)
    input_dim = tasks[0]["X_train"].shape[1]
    model = Finetune(
        input_dim, device=DEVICE, bert_mode=_BERT_MODE, bert_lr=bert_lr,
    )

    tasks_seen = []
    for task in tasks:
        X_tr, y_tr, _ = _subsample(task, _tune_sample_size())
        seen_now = _seen_classes_after(len(tasks_seen) + 1)
        model.train_task(X_tr, y_tr, epochs=_tune_epochs(), seen_classes=seen_now)
        tasks_seen.append(task)

    seen = _seen_classes_after(len(tasks_seen))
    score = _score(tasks_seen, _masked_predict(model.predict, seen))
    print(f"  [Fine-tune] bert_lr={bert_lr:.1e} | score={score:.4f}")
    return score


def _tune_group_dro(trial, tasks):
    eta      = trial.suggest_float("eta",     1e-3, 0.05, log=True)  # narrowed upper bound for BERT
    adj      = trial.suggest_float("adj",     0.0,  3.0)
    bert_lr  = trial.suggest_float("bert_lr", 5e-6, 5e-5, log=True) if _BERT_MODE else BIOS_BERT_LR

    input_dim = tasks[0]["X_train"].shape[1]
    model = GroupDRO(input_dim, eta=eta, adj=adj, device=DEVICE, bert_mode=_BERT_MODE)
    _patch_bert_lr(model, bert_lr)

    tasks_seen = []
    for task in tasks:
        X_tr, y_tr, s_tr = _subsample(task, _tune_sample_size())
        seen_now = _seen_classes_after(len(tasks_seen) + 1)
        model.train_task(X_tr, y_tr, s_tr, epochs=_tune_epochs(), seen_classes=seen_now)
        tasks_seen.append(task)

    seen = _seen_classes_after(len(tasks_seen))
    return _score(tasks_seen, _masked_predict(model.predict, seen))


def _tune_derpp(trial, tasks):
    alpha    = trial.suggest_float("alpha",   0.1, 2.0, log=True)
    beta     = trial.suggest_float("beta",    0.1, 2.0, log=True)
    bert_lr  = trial.suggest_float("bert_lr", 5e-6, 5e-5, log=True) if _BERT_MODE else BIOS_BERT_LR

    input_dim = tasks[0]["X_train"].shape[1]
    model = DERpp(input_dim, device=DEVICE, alpha=alpha, beta=beta, bert_mode=_BERT_MODE)
    _patch_bert_lr(model, bert_lr)

    tasks_seen = []
    for task in tasks:
        X_tr, y_tr, s_tr = _subsample(task, _tune_sample_size())
        seen_now = _seen_classes_after(len(tasks_seen) + 1)
        model.train_task(X_tr, y_tr, s_tr, epochs=_tune_epochs(), seen_classes=seen_now)
        model.remember(X_tr, y_tr, s_tr)
        tasks_seen.append(task)

    seen = _seen_classes_after(len(tasks_seen))
    score = _score(tasks_seen, _masked_predict(model.predict, seen))
    print(f"  [DER++] alpha={alpha:.3f} beta={beta:.3f} bert_lr={bert_lr:.1e} | score={score:.4f}")
    return score


def _tune_agem(trial, tasks):
    """
    A-GEM has no hyperparameters of its own once the memory budget is fixed by
    the parity constraint; on BERT its per-method encoder learning rate is
    tuned, as for every other method.
    """
    bert_lr = trial.suggest_float("bert_lr", 5e-6, 5e-5, log=True) if _BERT_MODE else BIOS_BERT_LR

    input_dim = tasks[0]["X_train"].shape[1]
    model = AGEM(input_dim, device=DEVICE, bert_mode=_BERT_MODE)
    _patch_bert_lr(model, bert_lr)

    tasks_seen = []
    for task in tasks:
        X_tr, y_tr, s_tr = _subsample(task, _tune_sample_size())
        seen_now = _seen_classes_after(len(tasks_seen) + 1)
        model.train_task(X_tr, y_tr, s_tr, epochs=_tune_epochs(), seen_classes=seen_now)
        model.remember(X_tr, y_tr, s_tr)
        tasks_seen.append(task)

    seen = _seen_classes_after(len(tasks_seen))
    score = _score(tasks_seen, _masked_predict(model.predict, seen))
    print(f"  [A-GEM] bert_lr={bert_lr:.1e} | score={score:.4f}")
    return score


def _tune_fairl(trial, tasks):
    lam     = trial.suggest_float("lam",     0.1, 5.0, log=True)
    eps     = trial.suggest_float("eps",     0.1, 2.0)
    adv_dim = trial.suggest_categorical("adv_dim", [32, 64, 128])
    bert_lr = trial.suggest_float("bert_lr", 5e-6, 5e-5, log=True) if _BERT_MODE else BIOS_BERT_LR

    input_dim = tasks[0]["X_train"].shape[1]
    model = FaIRL(input_dim, device=DEVICE, lam=lam, eps=eps, adv_dim=adv_dim, bert_mode=_BERT_MODE)
    _patch_bert_lr(model, bert_lr)

    tasks_seen = []
    for task in tasks:
        X_tr, y_tr, s_tr = _subsample(task, _tune_sample_size())
        seen_now = _seen_classes_after(len(tasks_seen) + 1)
        model.train_task(X_tr, y_tr, s_tr, epochs=_tune_epochs(), seen_classes=seen_now)
        model.remember(X_tr, y_tr, s_tr)
        tasks_seen.append(task)

    seen = _seen_classes_after(len(tasks_seen))
    score = _score(tasks_seen, _masked_predict(model.predict, seen))
    print(f"  [FaIRL] lam={lam:.3f} eps={eps:.3f} adv_dim={adv_dim} bert_lr={bert_lr:.1e} | score={score:.4f}")
    return score


def _tune_fsw(trial, tasks):
    alpha   = trial.suggest_float("alpha", 5e-4, 2e-2, log=True)
    lmbd    = trial.suggest_float("lmbd",  0.05, 2.0, log=True)
    tau     = trial.suggest_float("tau",   0.5, 10.0, log=True)
    bert_lr = trial.suggest_float("bert_lr", 5e-6, 5e-5, log=True) if _BERT_MODE else BIOS_BERT_LR

    input_dim = tasks[0]["X_train"].shape[1]
    model = FSW(input_dim, device=DEVICE, alpha=alpha, lmbd=lmbd, tau=tau, bert_mode=_BERT_MODE)
    _patch_bert_lr(model, bert_lr)

    tasks_seen = []
    for task in tasks:
        X_tr, y_tr, s_tr = _subsample(task, _tune_sample_size())
        seen_now = _seen_classes_after(len(tasks_seen) + 1)
        model.train_task(X_tr, y_tr, s_tr, epochs=_tune_epochs(), seen_classes=seen_now)
        model.remember(X_tr, y_tr, s_tr)
        tasks_seen.append(task)

    seen = _seen_classes_after(len(tasks_seen))
    score = _score(tasks_seen, _masked_predict(model.predict, seen))
    print(f"  [FSW] alpha={alpha:.4f} lmbd={lmbd:.3f} tau={tau:.2f} bert_lr={bert_lr:.1e} | score={score:.4f}")
    return score


def _tune_fairgem(trial, tasks):
    # alpha/beta are scale-invariant (g_safety is L2-normalised before projection)
    # so the same ranges apply for full-BERT and head-only modes.
    alpha    = trial.suggest_float("alpha",   0.05, 0.50)
    beta     = trial.suggest_float("beta",    0.50, 5.00)
    bert_lr  = trial.suggest_float("bert_lr", 5e-6, 5e-5, log=True) if _BERT_MODE else BIOS_BERT_LR

    input_dim = tasks[0]["X_train"].shape[1]
    model = FAIRGEM(input_dim, device=DEVICE, bert_mode=_BERT_MODE,
                    use_topk_dro=False, use_disparity=True, dynamic_eps=True,
                    alpha=alpha, beta=beta)
    _patch_bert_lr(model, bert_lr)

    proj_rates  = []
    tasks_seen  = []
    peak_scores = []

    for i, task in enumerate(tasks):
        X_tr, y_tr, s_tr = _subsample(task, _tune_sample_size())
        seen_now = _seen_classes_after(len(tasks_seen) + 1)
        model.train_task(X_tr, y_tr, s_tr, epochs=_tune_epochs(), seen_classes=seen_now)
        model.remember(X_tr, y_tr, s_tr)
        tasks_seen.append(task)
        peak_scores.append(_task_score(task, _masked_predict(model.predict, seen_now)))
        ps = model.proj_stats
        if i > 0:
            proj_rates.append(ps["safety_proj_rate"])

    # Diagnostics only. The objective is identical to every baseline's —
    # plain bal - EO_PENALTY * gap on validation.
    min_proj_rate = min(proj_rates) if proj_rates else 0.0

    seen = _seen_classes_after(len(tasks_seen))
    predict_masked = _masked_predict(model.predict, seen)
    final_scores = [_task_score(tasks[i], predict_masked) for i in range(len(tasks) - 1)]
    bwt = float(np.mean([max(0.0, peak_scores[i] - final_scores[i])
                         for i in range(len(tasks) - 1)]))

    score = _score(tasks_seen, predict_masked)
    print(f"  [FAIR-GEM] alpha={alpha:.3f} beta={beta:.3f} bert_lr={bert_lr:.1e} | "
          f"proj={min_proj_rate:.1%} bwt={bwt:.4f} | score={score:.4f}")
    return score


def _tune_fairderpp(trial, tasks):
    alpha     = trial.suggest_float("alpha",     0.1, 2.0, log=True)
    beta      = trial.suggest_float("beta",      0.1, 2.0, log=True)
    # Fairness projection margins: scale-invariant, same ranges as FAIR-GEM.
    eps_alpha = trial.suggest_float("eps_alpha", 0.05, 0.50)
    eps_beta  = trial.suggest_float("eps_beta",  0.50, 5.00)
    bert_lr   = trial.suggest_float("bert_lr",   5e-6, 5e-5, log=True) if _BERT_MODE else BIOS_BERT_LR

    input_dim = tasks[0]["X_train"].shape[1]
    model = FairDERpp(input_dim, device=DEVICE, bert_mode=_BERT_MODE,
                      alpha=alpha, beta=beta,
                      eps_alpha=eps_alpha, eps_beta=eps_beta,
                      disparity_mode=_disparity_mode())
    _patch_bert_lr(model, bert_lr)

    proj_rates = []
    tasks_seen = []
    for i, task in enumerate(tasks):
        X_tr, y_tr, s_tr = _subsample(task, _tune_sample_size())
        seen_now = _seen_classes_after(len(tasks_seen) + 1)
        model.train_task(X_tr, y_tr, s_tr, epochs=_tune_epochs(), seen_classes=seen_now)
        model.remember(X_tr, y_tr, s_tr)
        tasks_seen.append(task)
        ps = model.proj_stats
        if i > 0:
            proj_rates.append(ps["safety_proj_rate"])

    # Diagnostics only — see the note in _tune_fairgem.
    min_proj_rate = min(proj_rates) if proj_rates else 0.0

    seen = _seen_classes_after(len(tasks_seen))
    score = _score(tasks_seen, _masked_predict(model.predict, seen))
    print(f"  [Fair-DER++] alpha={alpha:.3f} beta={beta:.3f} "
          f"eps_alpha={eps_alpha:.3f} eps_beta={eps_beta:.3f} bert_lr={bert_lr:.1e} | "
          f"proj={min_proj_rate:.1%} | score={score:.4f}")
    return score


# Trial progress callback
def _make_callback(name, n_total):
    def callback(study, trial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        n_done = len([t for t in study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"  [{name:10s} {n_done:2d}/{n_total}]  "
              f"score={trial.value:.4f}  best={study.best_value:.4f}")
    return callback


# Config writer
def _fmt(v) -> str:
    if isinstance(v, int) or (isinstance(v, float) and v == int(v) and abs(v) < 1e9):
        return str(int(v))
    if v == 0.0:
        return "0.0"
    # Full precision: preflight.py compares config against the tuning
    # manifest at 5e-4 relative tolerance, which rounding would violate.
    return repr(float(v))


def _update_config(key: str, value: float):
    with open(CONFIG_PATH, encoding="utf-8") as f:
        text = f.read()

    pattern = rf"^({re.escape(key)}\s*=\s*)[\d.e+\-]+(.*)$"
    new_text, n = re.subn(pattern, rf"\g<1>{_fmt(value)}\2", text, flags=re.MULTILINE)

    if n == 0:
        print(f"  WARNING: {key!r} not found in config.py - skipping")
        return

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(new_text)
    print(f"  {key} = {_fmt(value)}")


def _write_tuning_summary(row: dict) -> None:
    """Upsert one completed study into the tuning manifest (tuning_summary.csv)."""
    fields = [
        "benchmark", "method", "tune_version", "objective", "trial_budget",
        "completed_trials", "best_score", "best_params", "memory_per_task",
    ]
    rows = []
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    rows = [
        old for old in rows
        if (old.get("benchmark"), old.get("method"))
        != (row["benchmark"], row["method"])
    ]
    rows.append(row)
    rows.sort(key=lambda item: item["method"])
    with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# Studies
# Equal trial budget for every method. A-GEM is included because FAIR-GEM is
# its direct extension.
N_TRIALS = 20

STUDIES = [
    # (display name, objective fn, n_trials, {CONFIG_KEY: param_name})
    ("Joint Retrain", _tune_joint_retrain, N_TRIALS, {
        "JOINT_BERT_LR": "bert_lr",
    }),
    ("Fine-tune",   _tune_finetune,     N_TRIALS, {
        "FINETUNE_BERT_LR": "bert_lr",
    }),
    ("A-GEM",      _tune_agem,       N_TRIALS, {
        "AGEM_BERT_LR":    "bert_lr",
    }),
    ("GroupDRO",   _tune_group_dro,  N_TRIALS, {
        "DRO_ETA":          "eta",
        "DRO_ADJ":          "adj",
        "GROUPDRO_BERT_LR": "bert_lr",   # written only when _BERT_MODE=True; per-method LR
    }),
    ("DER++",      _tune_derpp,      N_TRIALS, {
        "DERPP_ALPHA":     "alpha",
        "DERPP_BETA":      "beta",
        "DERPP_BERT_LR":   "bert_lr",
    }),
    ("FaIRL",      _tune_fairl,      N_TRIALS, {
        "FAIRL_LAMBDA":  "lam",
        "FAIRL_EPS":     "eps",
        "FAIRL_ADV_DIM": "adv_dim",
        "FAIRL_BERT_LR": "bert_lr",
    }),
    ("FSW",        _tune_fsw,        N_TRIALS, {
        "FSW_ALPHA":   "alpha",
        "FSW_LAMBDA":  "lmbd",
        "FSW_TAU":     "tau",
        "FSW_BERT_LR": "bert_lr",
    }),
    ("FAIR-GEM",   _tune_fairgem,    N_TRIALS, {
        "FAIRGEM_ALPHA":   "alpha",
        "FAIRGEM_BETA":    "beta",
        "FAIRGEM_BERT_LR": "bert_lr",
    }),
    ("Fair-DER++", _tune_fairderpp,  N_TRIALS, {
        "FAIRDERPP_ALPHA":     "alpha",
        "FAIRDERPP_BETA":      "beta",
        "FAIRDERPP_EPS_ALPHA": "eps_alpha",
        "FAIRDERPP_EPS_BETA":  "eps_beta",
        "FAIRDERPP_BERT_LR":   "bert_lr",
    }),
]


def _parse_tune_args(argv):
    all_names = [s[0] for s in STUDIES]
    methods_filter = None
    fresh = False

    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--methods" and i + 1 < len(argv):
            val = argv[i + 1]
            if val == "all":
                methods_filter = None
            else:
                requested = val.split(",")
                unknown = set(requested) - set(all_names)
                if unknown:
                    raise SystemExit(
                        f"Unknown study name(s): {sorted(unknown)}\n"
                        f"Valid names: {all_names}"
                    )
                methods_filter = requested
            i += 2
        elif arg == "--fresh":
            fresh = True
            i += 1
        else:
            i += 1

    return methods_filter, fresh


def run():
    import sys as _sys
    methods_filter, fresh = _parse_tune_args(_sys.argv)

    global _BERT_MODE


    active_studies = (
        STUDIES if methods_filter is None
        else [s for s in STUDIES if s[0] in methods_filter]
    )
    if not active_studies:
        raise SystemExit(f"No studies match --methods filter. Valid names: {[s[0] for s in STUDIES]}")

    print("Loading tune data...")
    tasks, _ = factory.get_tasks()
    tasks = tasks[:N_TUNE_TASKS]

    # Detect BERT mode from the loaded tasks.
    _BERT_MODE = factory.is_bert_task(tasks[0])
    n_samples  = BERT_N_TUNE_SAMPLES if _BERT_MODE else N_TUNE_SAMPLES

    print(f"  Tasks ({len(tasks)}): {[t['state'] for t in tasks]}")
    print(f"  BERT mode    : {_BERT_MODE}")
    print(f"  Subsampling to {n_samples:,} samples/task")
    print(f"  Tune version : {TUNE_VERSION}")
    print(f"  Objective    : bal - {EO_PENALTY} * gap  (identical for every method)")
    print(f"  Trials       : {N_TRIALS} per study")
    if methods_filter is not None:
        print(f"  Studies selected : {methods_filter}")
    print()

    storage_url = f"sqlite:///{DB_PATH}"

    for name, obj_fn, n_trials, key_map in active_studies:
        # Conditional-surrogate runs (CLFAIR_DISPARITY_MODE=conditional) use a
        # separate study and do not write back to config.py or the manifest.
        _cond = _disparity_mode() == "conditional"
        study_name = f"{name}_{TUNE_VERSION}" + ("_cond" if _cond else "")

        print(f"{'-'*60}")
        if fresh:
            try:
                optuna.delete_study(study_name=study_name, storage=storage_url)
                print(f"  [--fresh] deleted cached study {study_name!r}")
            except Exception:
                pass
        sampler = optuna.samplers.TPESampler(seed=SEED)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            storage=storage_url,
            study_name=study_name,
            load_if_exists=not fresh,
        )
        n_done = len([t for t in study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE])
        n_remaining = n_trials - n_done

        if n_remaining <= 0:
            print(f"Tuning {name}  - already complete ({n_done} trials cached)")
        else:
            print(f"Tuning {name}  ({n_remaining} trials remaining of {n_trials})...")
            study.optimize(
                lambda trial, fn=obj_fn: fn(trial, tasks),
                n_trials=n_remaining,
                show_progress_bar=False,
                callbacks=[_make_callback(name, n_trials)],
            )

        best  = study.best_params
        if _cond:
            print(f"  [conditional pilot] best={study.best_value:.4f} "
                  f"params={best} - NOT written to config.py or the manifest")
            continue
        score = study.best_value
        n_complete = len([
            trial for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ])
        print(f"  Best score : {score:.4f}")
        print(f"  Best params: {best}")
        print(f"  Writing to config.py:")
        for config_key, param_name in key_map.items():
            # bert_lr is only tuned in BERT mode; skip writing it otherwise
            # (the param won't exist in best if _BERT_MODE was False during tuning)
            if param_name not in best:
                continue
            _update_config(config_key, best[param_name])
        _write_tuning_summary({
            "benchmark": "bios",
            "method": name,
            "tune_version": TUNE_VERSION,
            "objective": f"bal_acc - {EO_PENALTY} * accuracy_parity_gap",
            "trial_budget": n_trials,
            "completed_trials": n_complete,
            "best_score": f"{score:.8g}",
            "best_params": json.dumps(best, sort_keys=True),
            "memory_per_task": MEMORY_SIZE_PER_TASK,
        })
        print()

    print("Done. Run:  python ../train.py --fresh")


if __name__ == "__main__":
    run()
