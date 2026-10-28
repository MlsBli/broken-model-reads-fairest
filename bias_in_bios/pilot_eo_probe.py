"""
pilot_eo_probe.py — Validation-EO readout for the conditional disparity surrogate.

The tuning composite (bal - 0.75 * AccParity gap) is dominated by balanced
accuracy on this benchmark, so a tied composite says little about EO, the
metric the conditional surrogate aligns with. This probe trains Fair-DER++ once
per surrogate mode under the tuning protocol (same subsample, epochs, masking)
using each mode's tuned parameters, and reports validation balanced accuracy,
AccParity gap, EO gap, and the composite. Validation data only; single fixed
seed.

Usage (run from bias_in_bios/results):  python ../pilot_eo_probe.py
Also exports the conditional study's per-trial record to pilot_conditional_trials.csv.
"""

import csv
import numpy as np
import torch

import factory
import tune
import config
from fairderpp import FairDERpp
from metrics import equalized_odds_gap, accuracy_parity_gap
from sklearn.metrics import balanced_accuracy_score

CONDITIONAL_BEST = dict(alpha=1.7347299696073073, beta=1.4954221394479263,
                        eps_alpha=0.4862916998545396, eps_beta=0.5808284402666883,
                        bert_lr=4.578145047518997e-05)
UNCONDITIONAL_BEST = dict(alpha=config.FAIRDERPP_ALPHA, beta=config.FAIRDERPP_BETA,
                          eps_alpha=config.FAIRDERPP_EPS_ALPHA,
                          eps_beta=config.FAIRDERPP_EPS_BETA,
                          bert_lr=config.FAIRDERPP_BERT_LR)


def run_mode(tasks, n_samples, mode, params):
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    input_dim = tasks[0]["X_train"].shape[1]
    model = FairDERpp(input_dim, device=config.DEVICE,
                      alpha=params["alpha"], beta=params["beta"],
                      eps_alpha=params["eps_alpha"], eps_beta=params["eps_beta"],
                      bert_mode=tune._BERT_MODE, disparity_mode=mode)
    tune._patch_bert_lr(model, params["bert_lr"])
    seen_all = None
    for k, task in enumerate(tasks):
        X, y, s = tune._subsample(task, n_samples)
        seen_all = tune._seen_classes_after(k + 1)
        model.train_task(X, y, s, epochs=tune._tune_epochs(), seen_classes=seen_all)
        model.remember(X, y, s)

    bals, aps, eos = [], [], []
    x_key = "token_ids_val" if tune._BERT_MODE else "X_val"
    predict = tune._masked_predict(model.predict, seen_all)
    for task in tasks:
        yv, sv = task["y_val"], task["s_val"]
        preds = predict(task[x_key])
        bals.append(balanced_accuracy_score(yv, preds))
        aps.append(accuracy_parity_gap(yv, preds, sv))
        eos.append(equalized_odds_gap(yv, preds, sv))
    bal, ap, eo = map(lambda v: float(np.nanmean(v)), (bals, aps, eos))
    comp = bal - tune.EO_PENALTY * ap
    print(f"{mode:>14}: BalAcc={bal:.4f}  AccParity={ap:.4f}  EO={eo:.4f}  composite={comp:.4f}")
    return dict(mode=mode, bal=bal, acc_parity=ap, eo=eo, composite=comp)


def export_trials():
    import optuna
    st = optuna.load_study(study_name=f"Fair-DER++_{tune.TUNE_VERSION}_cond",
                           storage=f"sqlite:///{tune.DB_PATH}")
    with open("pilot_conditional_trials.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trial", "state", "value", "params"])
        for t in st.trials:
            w.writerow([t.number, t.state.name, t.value, t.params])
    print(f"wrote pilot_conditional_trials.csv ({len(st.trials)} trials)")


def main():
    tasks, _ = factory.get_tasks()
    tasks = tasks[:tune.N_TUNE_TASKS]
    tune._BERT_MODE = factory.is_bert_task(tasks[0])
    n_samples = tune.BERT_N_TUNE_SAMPLES if tune._BERT_MODE else tune.N_TUNE_SAMPLES
    rows = [run_mode(tasks, n_samples, "unconditional", UNCONDITIONAL_BEST),
            run_mode(tasks, n_samples, "conditional", CONDITIONAL_BEST)]
    with open("pilot_eo_probe.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote pilot_eo_probe.csv")
    try:
        export_trials()
    except Exception as e:
        print(f"trial export skipped: {e}")


if __name__ == "__main__":
    main()
