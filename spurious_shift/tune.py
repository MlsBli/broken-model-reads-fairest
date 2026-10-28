"""
tune.py — Optuna hyperparameter search for the tunable methods.

Runs trials over all 6 tasks (subsampled to N_TUNE_SAMPLES each).

Objective per trial:  bal_acc - EO_PENALTY * accuracy_parity_gap.

Writes best values back to config.py.
"""

import csv
import json
import os
import re
import sys
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from metrics import accuracy_parity_gap
from baselines_neural import DERpp
from baselines_fairness import GroupDRO, FAIRL
from fairgem import FAIRGEM
from fairderpp import FairDERpp
from derpp_penalty import DERppPenalty
from fsw import FSW
from datasets.spurious_shift import get_spurious_tasks
from config import DEVICE, SEED, SPURIOUS_SEED, MEMORY_SIZE_PER_TASK

N_TUNE_TASKS   = 6
N_TUNE_SAMPLES = 20_000
CONFIG_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
DB_PATH        = "tune.db"
SUMMARY_PATH   = "tuning_summary.csv"
EO_PENALTY     = 0.75

# Suffix of the Optuna study names in tune.db; studies under a different
# suffix are not resumed.
TUNE_VERSION = "v8"

torch.manual_seed(SEED)
np.random.seed(SEED)


def _subsample(task, n):
    X, y, s = task["X_train"], task["y_train"], task["s_train"]
    if len(X) > n:
        idx = np.random.RandomState(SEED).choice(len(X), n, replace=False)
        return X[idx], y[idx], s[idx]
    return X, y, s


def _task_score(task, predict_fn):
    Xv, yv, sv = task["X_val"], task["y_val"], task["s_val"]
    preds = predict_fn(Xv)
    bal = float(balanced_accuracy_score(yv, preds))
    fair = float(accuracy_parity_gap(yv, preds, sv))
    if not np.isfinite(fair):
        # An undefined gap (a group under min_group_size) would drop the
        # fairness term from the selection objective, so fail instead.
        raise RuntimeError(
            f"accuracy_parity_gap undefined on val (n={len(yv)}, minority="
            f"{int(min(np.bincount(sv)))}); increase N_VAL or lower min_group_size")
    return bal - EO_PENALTY * fair


def _score(tasks_seen, predict_fn):
    return float(np.mean([_task_score(t, predict_fn) for t in tasks_seen]))


def _tune_group_dro(trial, tasks):
    eta = trial.suggest_float("eta", 1e-3, 0.1, log=True)
    adj = trial.suggest_float("adj", 0.0,  3.0)

    input_dim = tasks[0]["X_train"].shape[1]
    model = GroupDRO(input_dim, eta=eta, adj=adj, device=DEVICE)
    tasks_seen = []
    for task in tasks:
        X_tr, y_tr, s_tr = _subsample(task, N_TUNE_SAMPLES)
        model.train_task(X_tr, y_tr, s_tr)
        tasks_seen.append(task)
    return _score(tasks_seen, model.predict)


def _tune_derpp(trial, tasks):
    alpha = trial.suggest_float("alpha",    0.1, 2.0, log=True)
    beta = trial.suggest_float("beta",     0.1, 2.0, log=True)

    input_dim = tasks[0]["X_train"].shape[1]
    model = DERpp(input_dim, device=DEVICE, alpha=alpha, beta=beta)
    tasks_seen = []
    for task in tasks:
        X_tr, y_tr, s_tr = _subsample(task, N_TUNE_SAMPLES)
        model.train_task(X_tr, y_tr, s_tr)
        model.remember(X_tr, y_tr, s_tr)
        tasks_seen.append(task)

    score = _score(tasks_seen, model.predict)
    print(f"  [DER++] alpha={alpha:.3f} beta={beta:.3f} | score={score:.4f}")
    return score


def _tune_derpp_penalty(trial, tasks):
    fairness_weight = trial.suggest_float("fairness_weight", 1e-3, 10.0, log=True)
    input_dim = tasks[0]["X_train"].shape[1]
    model = DERppPenalty(input_dim, device=DEVICE, fairness_weight=fairness_weight)
    tasks_seen = []
    for task in tasks:
        X_tr, y_tr, s_tr = _subsample(task, N_TUNE_SAMPLES)
        model.train_task(X_tr, y_tr, s_tr)
        model.remember(X_tr, y_tr, s_tr)
        tasks_seen.append(task)
    score = _score(tasks_seen, model.predict)
    print(f"  [DER++-Penalty] lambda={fairness_weight:.4f} | score={score:.4f}")
    return score


def _tune_fairl(trial, tasks):
    lam = trial.suggest_float("lam", 0.1, 5.0, log=True)
    eps = trial.suggest_float("eps", 0.1, 2.0)
    adv_dim = trial.suggest_categorical("adv_dim", [32, 64, 128])

    input_dim = tasks[0]["X_train"].shape[1]
    model = FAIRL(input_dim, device=DEVICE, lam=lam, eps=eps, adv_dim=adv_dim)
    tasks_seen = []
    for task in tasks:
        X_tr, y_tr, s_tr = _subsample(task, N_TUNE_SAMPLES)
        model.train_task(X_tr, y_tr, s_tr)
        model.remember(X_tr, y_tr, s_tr)
        tasks_seen.append(task)

    score = _score(tasks_seen, model.predict)
    print(f"  [FAIRL] lam={lam:.3f} eps={eps:.3f} adv_dim={adv_dim} | score={score:.4f}")
    return score


def _tune_fsw(trial, tasks):
    alpha = trial.suggest_float("alpha", 5e-4, 2e-2, log=True)
    lmbd = trial.suggest_float("lmbd",  0.05, 2.0, log=True)
    tau = trial.suggest_float("tau",   0.5, 10.0, log=True)

    input_dim = tasks[0]["X_train"].shape[1]
    model = FSW(input_dim, device=DEVICE, alpha=alpha, lmbd=lmbd, tau=tau)
    tasks_seen = []
    for task in tasks:
        X_tr, y_tr, s_tr = _subsample(task, N_TUNE_SAMPLES)
        model.train_task(X_tr, y_tr, s_tr)
        model.remember(X_tr, y_tr, s_tr)
        tasks_seen.append(task)

    score = _score(tasks_seen, model.predict)
    print(f"  [FSW] alpha={alpha:.4f} lmbd={lmbd:.3f} tau={tau:.2f} | score={score:.4f}")
    return score


def _tune_fairgem(trial, tasks):
    alpha = trial.suggest_float("alpha",    0.05, 0.50)
    beta = trial.suggest_float("beta",     0.50, 5.00)

    input_dim = tasks[0]["X_train"].shape[1]
    model = FAIRGEM(input_dim, device=DEVICE, use_topk_dro=False, use_disparity=True,
                          dynamic_eps=True, alpha=alpha, beta=beta)
    proj_rates = []
    tasks_seen = []
    peak_scores = []

    for i, task in enumerate(tasks):
        X_tr, y_tr, s_tr = _subsample(task, N_TUNE_SAMPLES)
        model.train_task(X_tr, y_tr, s_tr)
        model.remember(X_tr, y_tr, s_tr)
        tasks_seen.append(task)
        peak_scores.append(_task_score(task, model.predict))
        if i > 0:
            proj_rates.append(model.proj_stats["safety_proj_rate"])

    # Diagnostics only. The objective is identical to every baseline's:
    # plain bal - EO_PENALTY * gap on validation.
    min_proj_rate = min(proj_rates) if proj_rates else 0.0
    final_scores = [_task_score(tasks[i], model.predict) for i in range(len(tasks) - 1)]
    bwt = float(np.mean([max(0.0, peak_scores[i] - final_scores[i])
                         for i in range(len(tasks) - 1)]))

    score = _score(tasks_seen, model.predict)
    print(f"  [FAIR-GEM] alpha={alpha:.3f} beta={beta:.3f} | "
          f"proj={min_proj_rate:.1%} bwt={bwt:.4f} | score={score:.4f}")
    return score


def _tune_fairderpp(trial, tasks):
    alpha = trial.suggest_float("alpha",     0.1, 2.0, log=True)
    beta = trial.suggest_float("beta",      0.1, 2.0, log=True)
    eps_alpha = trial.suggest_float("eps_alpha", 0.05, 0.50)
    eps_beta = trial.suggest_float("eps_beta",  0.50, 5.00)

    input_dim = tasks[0]["X_train"].shape[1]
    model = FairDERpp(input_dim, device=DEVICE, alpha=alpha, beta=beta,
                           eps_alpha=eps_alpha, eps_beta=eps_beta)
    proj_rates = []
    tasks_seen = []
    for i, task in enumerate(tasks):
        X_tr, y_tr, s_tr = _subsample(task, N_TUNE_SAMPLES)
        model.train_task(X_tr, y_tr, s_tr)
        model.remember(X_tr, y_tr, s_tr)
        tasks_seen.append(task)
        if i > 0:
            proj_rates.append(model.proj_stats["safety_proj_rate"])

    # Diagnostics only — see the note in _tune_fairgem.
    min_proj_rate = min(proj_rates) if proj_rates else 0.0
    score = _score(tasks_seen, model.predict)
    print(f"  [Fair-DER++] alpha={alpha:.3f} beta={beta:.3f} eps_alpha={eps_alpha:.3f} eps_beta={eps_beta:.3f} "
          f"| proj={min_proj_rate:.1%} | score={score:.4f}")
    return score


def _make_callback(name, n_total):
    def callback(study, trial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        n_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"  [{name:10s} {n_done:2d}/{n_total}]  score={trial.value:.4f}  best={study.best_value:.4f}")
    return callback


def _fmt(v):
    if isinstance(v, int) or (isinstance(v, float) and v == int(v) and abs(v) < 1e9):
        return str(int(v))
    if v == 0.0:
        return "0.0"
    # Full precision, so the written config reproduces the tuned values
    # exactly.
    return repr(float(v))


def _update_config(key, value):
    with open(CONFIG_PATH, encoding="utf-8") as f:
        text = f.read()
    pattern  = rf"^({re.escape(key)}\s*=\s*)[\d.e+\-]+(.*)$"
    new_text, n = re.subn(pattern, rf"\g<1>{_fmt(value)}\2", text, flags=re.MULTILINE)
    if n == 0:
        print(f"  WARNING: {key!r} not found in config.py - skipping")
        return
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(new_text)
    print(f"  {key} = {_fmt(value)}")


def _write_tuning_summary(row):
    """Upsert one completed study into the tuning summary CSV."""
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


# Equal trial budget for every method.
#
# A-GEM has no study: with the memory budget fixed across methods it has no
# free hyperparameters (the learning rate is shared via MLP_LR), so it is fully
# determined by the shared training configuration. (On Bias in Bios it has a
# tunable per-method BERT learning rate and gets a study.)
N_TRIALS = 15

def _tune_fairgem_sgd(trial, tasks):
    lr = trial.suggest_float("lr", 1e-4, 3e-2, log=True)
    input_dim = tasks[0]["X_train"].shape[1]
    model = FAIRGEM(input_dim, device=DEVICE, use_topk_dro=False, use_disparity=True,
                    dynamic_eps=True, opt="sgd", opt_lr=lr)
    tasks_seen = []
    for task in tasks:
        X_tr, y_tr, s_tr = _subsample(task, N_TUNE_SAMPLES)
        model.train_task(X_tr, y_tr, s_tr)
        model.remember(X_tr, y_tr, s_tr)
        tasks_seen.append(task)
    score = _score(tasks_seen, model.predict)
    print(f"  [FAIR-GEM-SGD] lr={lr:.2e} | score={score:.4f}")
    return score


def _tune_fairderpp_sgd(trial, tasks):
    lr = trial.suggest_float("lr", 1e-4, 3e-2, log=True)
    input_dim = tasks[0]["X_train"].shape[1]
    model = FairDERpp(input_dim, device=DEVICE, opt="sgd", opt_lr=lr)
    tasks_seen = []
    for task in tasks:
        X_tr, y_tr, s_tr = _subsample(task, N_TUNE_SAMPLES)
        model.train_task(X_tr, y_tr, s_tr)
        model.remember(X_tr, y_tr, s_tr)
        tasks_seen.append(task)
    score = _score(tasks_seen, model.predict)
    print(f"  [Fair-DER++-SGD] lr={lr:.2e} | score={score:.4f}")
    return score


STUDIES = [
    ("GroupDRO", _tune_group_dro, N_TRIALS, {"DRO_ETA": "eta", "DRO_ADJ": "adj"}),
    ("DER++", _tune_derpp, N_TRIALS, {"DERPP_ALPHA": "alpha", "DERPP_BETA": "beta"}),
    ("DER++-Penalty", _tune_derpp_penalty, N_TRIALS,
     {"DERPP_FAIR_LAMBDA": "fairness_weight"}),
    ("FAIRL", _tune_fairl, N_TRIALS, {"FAIRL_LAMBDA": "lam", "FAIRL_EPS": "eps",
                                          "FAIRL_ADV_DIM": "adv_dim"}),
    ("FAIR-GEM-SGD", _tune_fairgem_sgd, N_TRIALS, {"FAIRGEM_SGD_LR": "lr"}),
    ("Fair-DER++-SGD", _tune_fairderpp_sgd, N_TRIALS, {"FAIRDERPP_SGD_LR": "lr"}),
    ("FSW", _tune_fsw, N_TRIALS, {"FSW_ALPHA": "alpha", "FSW_LAMBDA": "lmbd",
                                          "FSW_TAU": "tau"}),
    ("FAIR-GEM", _tune_fairgem, N_TRIALS, {"FAIRGEM_ALPHA": "alpha", "FAIRGEM_BETA": "beta"}),
    ("Fair-DER++", _tune_fairderpp, N_TRIALS, {"FAIRDERPP_ALPHA": "alpha", "FAIRDERPP_BETA": "beta",
                                          "FAIRDERPP_EPS_ALPHA": "eps_alpha", "FAIRDERPP_EPS_BETA": "eps_beta"}),
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
                unknown   = set(requested) - set(all_names)
                if unknown:
                    raise SystemExit(f"Unknown study name(s): {sorted(unknown)}\nValid: {all_names}")
                methods_filter = requested
            i += 2
        elif arg == "--fresh":
            fresh = True; i += 1
        else:
            i += 1

    return methods_filter, fresh


def run():
    methods_filter, fresh = _parse_tune_args(sys.argv)

    active_studies = (STUDIES if methods_filter is None
                      else [s for s in STUDIES if s[0] in methods_filter])
    if not active_studies:
        raise SystemExit(f"No studies match filter. Valid: {[s[0] for s in STUDIES]}")

    print("Loading tune data...")
    tasks, _ = get_spurious_tasks(seed=SPURIOUS_SEED)
    tasks = tasks[:N_TUNE_TASKS]
    print(f"  Tasks ({len(tasks)}): {[t['state'] for t in tasks]}")
    print(f"  Subsampling to {N_TUNE_SAMPLES:,} samples/task")
    print(f"  Tune version: {TUNE_VERSION}")
    print(f"  Objective:    bal - {EO_PENALTY} * gap  (identical for every method)")
    print(f"  Trials:       {N_TRIALS} per study")
    if methods_filter is not None:
        print(f"  Studies: {methods_filter}")
    print()

    storage_url = f"sqlite:///{DB_PATH}"

    for name, obj_fn, n_trials, key_map in active_studies:
        study_name = f"{name}_{TUNE_VERSION}"

        print(f"{'-'*60}")
        if fresh:
            try:
                optuna.delete_study(study_name=study_name, storage=storage_url)
                print(f"  [--fresh] deleted {study_name!r}")
            except Exception:
                pass
        sampler = optuna.samplers.TPESampler(seed=SEED)
        study   = optuna.create_study(direction="maximize", sampler=sampler,
                                      storage=storage_url, study_name=study_name,
                                      load_if_exists=not fresh)
        n_done      = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_remaining = n_trials - n_done

        if n_remaining <= 0:
            print(f"Tuning {name} - already complete ({n_done} trials cached)")
        else:
            print(f"Tuning {name} ({n_remaining} trials remaining of {n_trials})...")
            study.optimize(lambda trial, fn=obj_fn: fn(trial, tasks),
                           n_trials=n_remaining, show_progress_bar=False,
                           callbacks=[_make_callback(name, n_trials)])

        best = study.best_params
        n_complete = len([
            trial for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ])
        print(f"  Best score: {study.best_value:.4f}")
        print(f"  Best params: {best}")
        print(f"  Writing to config.py:")
        for config_key, param_name in key_map.items():
            _update_config(config_key, best[param_name])
        _write_tuning_summary({
            "benchmark": "spurious",
            "method": name,
            "tune_version": TUNE_VERSION,
            "objective": f"bal_acc - {EO_PENALTY} * accuracy_parity_gap",
            "trial_budget": n_trials,
            "completed_trials": n_complete,
            "best_score": f"{study.best_value:.8g}",
            "best_params": json.dumps(best, sort_keys=True),
            "memory_per_task": MEMORY_SIZE_PER_TASK,
        })
        print()

    print("Done. Run:  python ../train.py --fresh")


if __name__ == "__main__":
    run()
