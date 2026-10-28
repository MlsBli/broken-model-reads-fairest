"""
run_ablations.py - Train the Spurious Shift ablation variants (see the
ablations in the appendix).

Run from spurious_shift/results:
    python ../run_ablations.py --fresh --seed 0

Output:
    results_ablations.npz
"""

import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

import config as cfg
import runctx
from config import DEVICE, SPURIOUS_N_TASKS
from datasets.spurious_shift import get_spurious_tasks
from fairderpp import FairDERpp
from derpp_penalty import DERppPenalty
from baselines_neural import AGEM, DERpp
from fairgem import FAIRGEM
from metrics import (
    accuracy_parity_gap,
    task_accuracy,
    top_class_rate_gap,
)


VARIANTS = [
    "fair_gem_cycles_1",
    "fair_gem_cycles_3",
    "fair_gem_cycles_5",
    "fair_gem_cycles_30",
    "agem_mem200", "agem_mem1600",
    "fair_gem_mem200", "fair_gem_mem1600",
    "derpp_mem200", "derpp_mem1600",
    "fair_derpp_mem200", "fair_derpp_mem1600",
    "fair_gem_sgd_proj",
    "fair_derpp_sgd_proj",
    "fair_gem_static_eps",
    "fair_gem_avg_grad",
    "fair_gem_topk_dro",
    "fair_gem_uniform_memory",
    "fair_gem_alpha_low",
    "fair_gem_alpha_high",
    "fair_gem_beta_low",
    "fair_gem_beta_high",
    "fair_derpp_static_eps",
    "fair_derpp_avg_grad",
    "fair_derpp_uniform_memory",
    "fair_derpp_alpha_low",
    "fair_derpp_alpha_high",
    "fair_derpp_beta_low",
    "fair_derpp_beta_high",
    "derpp_loss_penalty",
]

# --seed N mirrors train.py: seeds the RNGs and writes artifacts to
# runs/seedN/.
SEED_ARG       = runctx.parse_seed(sys.argv)
EFFECTIVE_SEED = runctx.apply_seed(SEED_ARG)
DATA_SEED      = runctx.data_seed(SEED_ARG)
OUT            = runctx.output_dir(SEED_ARG)

CHECKPOINT_PATH = os.path.join(OUT, "checkpoint_ablations.pt")
RESULTS_PATH = os.path.join(OUT, "results_ablations.npz")
N_TASKS = SPURIOUS_N_TASKS



def _make_variants(input_dim, mean_eps_fairgem, mean_eps_fairderpp):
    def uniform_memory(model):
        model.memory.stratified = False
        return model

    return {
        "fair_gem_cycles_1": FAIRGEM(input_dim, device=DEVICE, use_topk_dro=False, use_disparity=True, dynamic_eps=True, proj_iters=1),
        "fair_gem_cycles_3": FAIRGEM(input_dim, device=DEVICE, use_topk_dro=False, use_disparity=True, dynamic_eps=True, proj_iters=3),
        "fair_gem_cycles_5": FAIRGEM(input_dim, device=DEVICE, use_topk_dro=False, use_disparity=True, dynamic_eps=True, proj_iters=5),
        "fair_gem_cycles_30": FAIRGEM(input_dim, device=DEVICE, use_topk_dro=False, use_disparity=True, dynamic_eps=True, proj_iters=30),
        "agem_mem200": AGEM(input_dim, device=DEVICE, mem_size=200),
        "fair_gem_mem200": FAIRGEM(input_dim, device=DEVICE, use_topk_dro=False, use_disparity=True, dynamic_eps=True, mem_size=200),
        "derpp_mem200": DERpp(input_dim, device=DEVICE, mem_size=200),
        "fair_derpp_mem200": FairDERpp(input_dim, device=DEVICE, mem_size=200),
        "agem_mem1600": AGEM(input_dim, device=DEVICE, mem_size=1600),
        "fair_gem_mem1600": FAIRGEM(input_dim, device=DEVICE, use_topk_dro=False, use_disparity=True, dynamic_eps=True, mem_size=1600),
        "derpp_mem1600": DERpp(input_dim, device=DEVICE, mem_size=1600),
        "fair_derpp_mem1600": FairDERpp(input_dim, device=DEVICE, mem_size=1600),
        "fair_gem_sgd_proj": FAIRGEM(
            input_dim,
            device=DEVICE,
            use_topk_dro=False,
            use_disparity=True,
            dynamic_eps=True,
            opt="sgd",
            opt_lr=cfg.FAIRGEM_SGD_LR,
        ),
        "fair_derpp_sgd_proj": FairDERpp(
            input_dim,
            device=DEVICE,
            opt="sgd",
            opt_lr=cfg.FAIRDERPP_SGD_LR,
        ),
        "fair_gem_static_eps": FAIRGEM(
            input_dim,
            device=DEVICE,
            use_topk_dro=False,
            use_disparity=True,
            dynamic_eps=False,
            alpha=mean_eps_fairgem,
        ),
        "fair_gem_avg_grad": FAIRGEM(
            input_dim,
            device=DEVICE,
            use_topk_dro=False,
            use_disparity=False,
            dynamic_eps=True,
        ),
        "fair_gem_topk_dro": FAIRGEM(
            input_dim,
            device=DEVICE,
            use_topk_dro=True,
            use_disparity=True,
            dynamic_eps=True,
        ),
        "fair_gem_uniform_memory": uniform_memory(FAIRGEM(
            input_dim,
            device=DEVICE,
            use_topk_dro=False,
            use_disparity=True,
            dynamic_eps=True,
        )),
        "fair_gem_alpha_low": FAIRGEM(
            input_dim, device=DEVICE, alpha=0.75 * cfg.FAIRGEM_ALPHA,
            beta=cfg.FAIRGEM_BETA,
        ),
        "fair_gem_alpha_high": FAIRGEM(
            input_dim, device=DEVICE, alpha=1.25 * cfg.FAIRGEM_ALPHA,
            beta=cfg.FAIRGEM_BETA,
        ),
        "fair_gem_beta_low": FAIRGEM(
            input_dim, device=DEVICE, alpha=cfg.FAIRGEM_ALPHA,
            beta=0.75 * cfg.FAIRGEM_BETA,
        ),
        "fair_gem_beta_high": FAIRGEM(
            input_dim, device=DEVICE, alpha=cfg.FAIRGEM_ALPHA,
            beta=1.25 * cfg.FAIRGEM_BETA,
        ),
        "fair_derpp_static_eps": FairDERpp(
            input_dim,
            device=DEVICE,
            dynamic_eps=False,
            eps_alpha=mean_eps_fairderpp,
        ),
        "fair_derpp_avg_grad": FairDERpp(
            input_dim,
            device=DEVICE,
            use_disparity=False,
            dynamic_eps=True,
        ),
        "fair_derpp_uniform_memory": uniform_memory(FairDERpp(
            input_dim,
            device=DEVICE,
            dynamic_eps=True,
        )),
        "fair_derpp_alpha_low": FairDERpp(
            input_dim, device=DEVICE,
            eps_alpha=0.75 * cfg.FAIRDERPP_EPS_ALPHA,
            eps_beta=cfg.FAIRDERPP_EPS_BETA,
        ),
        "fair_derpp_alpha_high": FairDERpp(
            input_dim, device=DEVICE,
            eps_alpha=1.25 * cfg.FAIRDERPP_EPS_ALPHA,
            eps_beta=cfg.FAIRDERPP_EPS_BETA,
        ),
        "fair_derpp_beta_low": FairDERpp(
            input_dim, device=DEVICE,
            eps_alpha=cfg.FAIRDERPP_EPS_ALPHA,
            eps_beta=0.75 * cfg.FAIRDERPP_EPS_BETA,
        ),
        "fair_derpp_beta_high": FairDERpp(
            input_dim, device=DEVICE,
            eps_alpha=cfg.FAIRDERPP_EPS_ALPHA,
            eps_beta=1.25 * cfg.FAIRDERPP_EPS_BETA,
        ),
        "derpp_loss_penalty": DERppPenalty(
            input_dim,
            device=DEVICE,
            fairness_weight=cfg.DERPP_FAIR_LAMBDA,
        ),
    }


def _state_for(models):
    return {
        name: {
            "model": obj.model.state_dict(),
            "memory": obj.memory,
        }
        for name, obj in models.items()
    }


def _restore(models, state):
    for name, obj in models.items():
        if name not in state:
            continue
        obj.model.load_state_dict(state[name]["model"])
        obj.memory = state[name]["memory"]


def _calibrate_eps(tasks, input_dim):
    print("Calibrating mean dynamic eps_t for static-epsilon ablations...")
    fairgem = FAIRGEM(input_dim, device=DEVICE)
    fairderpp = FairDERpp(input_dim, device=DEVICE)
    eps_fairgem = []
    eps_fairderpp = []

    for t_idx, task in enumerate(tasks):
        fairgem.train_task(task["X_train"], task["y_train"], task["s_train"])
        fairgem.remember(task["X_train"], task["y_train"], task["s_train"])
        eps_fairgem.append(fairgem.proj_stats["mean_eps_t"])

        fairderpp.train_task(task["X_train"], task["y_train"], task["s_train"])
        fairderpp.remember(task["X_train"], task["y_train"], task["s_train"])
        eps_fairderpp.append(fairderpp.proj_stats["mean_eps_t"])

        print(
            f"  task {t_idx + 1}/{len(tasks)} "
            f"fair_gem={eps_fairgem[-1]:.4f} "
            f"fair_derpp={eps_fairderpp[-1]:.4f}"
        )

    return float(np.mean(eps_fairgem)), float(np.mean(eps_fairderpp))


def run():
    fresh = "--fresh" in sys.argv
    # --arms a,b restricts this run to a subset of VARIANTS; the results file
    # then contains only those arms, so it should not be aggregated with full
    # runs.
    arms = None
    for i, a in enumerate(sys.argv):
        if a == "--arms" and i + 1 < len(sys.argv):
            arms = sys.argv[i + 1].split(",")
            unknown = set(arms) - set(VARIANTS)
            if unknown:
                raise SystemExit(f"Unknown arm(s): {sorted(unknown)}")
    if fresh and os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
    if fresh and os.path.exists(RESULTS_PATH):
        os.remove(RESULTS_PATH)
    if not fresh and os.path.exists(RESULTS_PATH):
        print(f"{RESULTS_PATH} already exists. Use --fresh to retrain.")
        return

    tasks, _ = get_spurious_tasks(seed=DATA_SEED)
    input_dim = tasks[0]["X_train"].shape[1]

    acc = {m: np.full((N_TASKS, N_TASKS), np.nan) for m in VARIANTS}
    bal_acc = {m: np.full((N_TASKS, N_TASKS), np.nan) for m in VARIANTS}
    eo = {m: np.full((N_TASKS, N_TASKS), np.nan) for m in VARIANTS}
    dp = {m: np.full((N_TASKS, N_TASKS), np.nan) for m in VARIANTS}

    start_task = 0
    mean_eps_fairgem = None
    mean_eps_fairderpp = None
    if not fresh and os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, weights_only=False)
        acc = ckpt["acc"]
        bal_acc = ckpt["bal_acc"]
        eo = ckpt["eo"]
        dp = ckpt["dp"]
        start_task = ckpt["last_completed_task"] + 1
        mean_eps_fairgem = ckpt["mean_eps_fairgem"]
        mean_eps_fairderpp = ckpt["mean_eps_fairderpp"]
        print(f"Resuming ablations from task {start_task + 1}/{N_TASKS}")

    if mean_eps_fairgem is None or mean_eps_fairderpp is None:
        mean_eps_fairgem, mean_eps_fairderpp = _calibrate_eps(tasks, input_dim)
        print(
            "Mean dynamic eps_t: "
            f"fair_gem={mean_eps_fairgem:.4f}, "
            f"fair_derpp={mean_eps_fairderpp:.4f}"
        )

    models = _make_variants(input_dim, mean_eps_fairgem, mean_eps_fairderpp)
    if arms is not None:
        models = {k: v for k, v in models.items() if k in arms}
        print(f"  --arms subset: {sorted(models)}")
    if not fresh and os.path.exists(CHECKPOINT_PATH):
        _restore(models, ckpt["state"])

    for t_idx in range(start_task, N_TASKS):
        task = tasks[t_idx]
        X_tr = task["X_train"]
        y_tr = task["y_train"]
        s_tr = task["s_train"]
        print(f"\nTask {t_idx + 1}/{N_TASKS}: {task['state']}")

        for name, model in models.items():
            t0 = time.time()
            model.train_task(X_tr, y_tr, s_tr)
            model.remember(X_tr, y_tr, s_tr)
            print(f"  {name:<24} {time.time() - t0:.1f}s")

        for e_idx in range(t_idx + 1):
            et = tasks[e_idx]
            yv = et["y_test"]
            sv = et["s_test"]
            for name, model in models.items():
                preds = model.predict(et["X_test"])
                acc[name][t_idx, e_idx] = task_accuracy(yv, preds)
                bal_acc[name][t_idx, e_idx] = balanced_accuracy_score(yv, preds)
                eo[name][t_idx, e_idx] = accuracy_parity_gap(yv, preds, sv)
                dp[name][t_idx, e_idx] = top_class_rate_gap(preds, sv)

        torch.save(
            {
                "last_completed_task": t_idx,
                "mean_eps_fairgem": mean_eps_fairgem,
                "mean_eps_fairderpp": mean_eps_fairderpp,
                "acc": acc,
                "bal_acc": bal_acc,
                "eo": eo,
                "dp": dp,
                "state": _state_for(models),
            },
            CHECKPOINT_PATH,
        )

    task_names = [t["state"] for t in tasks]
    np.savez(
        RESULTS_PATH,
        acc=acc,
        bal_acc=bal_acc,
        eo=eo,
        dp=dp,
        methods=VARIANTS,
        states=task_names,
        mean_eps_fairgem=mean_eps_fairgem,
        mean_eps_fairderpp=mean_eps_fairderpp,
    )
    print(f"\nSaved {RESULTS_PATH}")


if __name__ == "__main__":
    run()
