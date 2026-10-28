"""
train.py — Train all methods sequentially and save results.

Run from spurious_shift/results:
    python ../train.py --fresh --seed 0
    python ../train.py --reeval --seed 0

--fresh to restart training
--reeval to write the final-row test view (results_test.npz)
--methods novel for FAIR-GEM and Fair-DER++ only
--methods baselines for the baselines only
--seed N to run one seed of a sweep; artifacts go to runs/seedN/ instead of
    the current directory, and N seeds the model init, batch shuffling, and
    the generated dataset. Without --seed, cfg.SEED is used and artifacts are
    written to the current directory.
"""

import os
import sys
import numpy as np
import torch
import time
from sklearn.metrics import balanced_accuracy_score

from metrics import (group_loss_disparity, conditional_group_loss_disparity,
                     task_accuracy, accuracy_parity_gap, top_class_rate_gap,
                     equalized_odds_gap, summarize, per_group_accuracy)
from baselines_neural import JointRetrain, Finetune, AGEM, DERpp
from baselines_fairness import GroupDRO, FAIRL
from fairgem import FAIRGEM
from fairderpp import FairDERpp
from fsw import FSW
from model import per_sample_ce
import runctx
from datasets.spurious_shift import get_spurious_tasks
from config import DEVICE, SPURIOUS_N_TASKS

# --seed N seeds the RNGs *and* writes every artifact to runs/seedN/. Without
# --seed: cfg.SEED, writing to the current directory.
SEED_ARG       = runctx.parse_seed(sys.argv)
EFFECTIVE_SEED = runctx.apply_seed(SEED_ARG)
DATA_SEED      = runctx.data_seed(SEED_ARG)
OUT            = runctx.output_dir(SEED_ARG)

METHODS = ["joint_retrain", "finetune", "agem", "der_pp", "group_dro", "fairl", "fsw", "fair_gem", "fair_derpp"]
BASELINE_METHODS = ["joint_retrain", "finetune", "agem", "der_pp", "group_dro", "fairl", "fsw"]
NOVEL_METHODS = ["fair_gem", "fair_derpp"]

# Reported alongside the headline metrics.
#   eo_true      textbook equalized-odds gap (max TPR/FPR gap across groups).
#                The array named 'eo' holds the *accuracy-parity* gap, the
#                paper's headline fairness metric; keeping both lets them be
#                compared directly.
#   worst_group  worst per-group balanced accuracy, so a low disparity reached
#                by collapsing to chance is visible.
EXTRA_METRICS = ["eo_true", "worst_group", "loss_gap", "loss_gap_cond"]

N_TASKS = SPURIOUS_N_TASKS
CHECKPOINT_PATH = os.path.join(OUT, "checkpoint.pt")
BASELINE_CHECKPOINT_PATH = os.path.join(OUT, "baseline_checkpoint.pt")
RESULTS_PATH = os.path.join(OUT, "results.npz")
RESULTS_GROUPS_PATH = os.path.join(OUT, "results_groups.npy")
RESULTS_TEST_PATH = os.path.join(OUT, "results_test.npz")
RESULTS_TEST_GROUPS_PATH = os.path.join(OUT, "results_test_groups.npy")
PROJ_STATS_PATH = os.path.join(OUT, "results_proj_stats.npy")


def _parse_methods(argv):
    if "--baselines-only" in argv and "--methods" not in argv:
        return set(BASELINE_METHODS)
    if "--novel-only" in argv and "--methods" not in argv:
        return set(NOVEL_METHODS)

    for i, arg in enumerate(argv):
        if arg == "--methods" and i + 1 < len(argv):
            val = argv[i + 1]
            if val == "all":       return set(METHODS)
            if val == "baselines": return set(BASELINE_METHODS)
            if val == "novel":     return set(NOVEL_METHODS)
            requested = set(val.split(","))
            unknown   = requested - set(METHODS)
            if unknown:
                raise SystemExit(f"Unknown method(s): {sorted(unknown)}\nValid: {METHODS}")
            return requested

    return set(METHODS)


def _collect_state(joint_retrain, finetune, agem, der_pp, group_dro, fairl, fsw, fair_gem, fair_derpp):
    state = {
        'joint_retrain_model': joint_retrain.model.state_dict(),
        'finetune_model': finetune.model.state_dict(),
        'agem_model': agem.model.state_dict(),
        'agem_memory': agem.memory,
        'der_pp_model': der_pp.model.state_dict(),
        'der_pp_memory': der_pp.memory,
        'group_dro_model': group_dro.model.state_dict(),
        'fairl_model': fairl.model.state_dict(),
        'fairl_memory': fairl.memory,
        'fsw_model': fsw.model.state_dict(),
        'fsw_memory': fsw.memory,
        'fair_gem_model': fair_gem.model.state_dict(),
        'fair_gem_memory': fair_gem.memory,
        'fair_derpp_model': fair_derpp.model.state_dict(),
        'fair_derpp_memory': fair_derpp.memory,
    }
    if hasattr(joint_retrain, 'all_X'):
        state['joint_retrain_data'] = (joint_retrain.all_X, joint_retrain.all_y)
    return state


def _restore_state(m, joint_retrain, finetune, agem, der_pp, group_dro, fairl, fsw, fair_gem, fair_derpp):
    joint_retrain.model.load_state_dict(m['joint_retrain_model'])
    if 'joint_retrain_data' in m and hasattr(joint_retrain, 'all_X'):
        joint_retrain.all_X, joint_retrain.all_y = m['joint_retrain_data']
    finetune.model.load_state_dict(m['finetune_model'])
    agem.model.load_state_dict(m['agem_model'])
    agem.memory = m['agem_memory']
    if 'der_pp_model' in m:
        der_pp.model.load_state_dict(m['der_pp_model'])
        der_pp.memory = m['der_pp_memory']
    else:
        print("  WARNING: checkpoint predates DER++ - der_pp results will be NaN")
    if 'group_dro_model' in m:
        group_dro.model.load_state_dict(m['group_dro_model'])
    else:
        print("  WARNING: checkpoint predates GroupDRO - group_dro results will be NaN")
    if 'fairl_model' in m:
        fairl.model.load_state_dict(m['fairl_model'])
        fairl.memory = m['fairl_memory']
    else:
        print("  WARNING: checkpoint predates FAIRL - fairl results will be NaN")
    if 'fsw_model' in m:
        fsw.model.load_state_dict(m['fsw_model'])
        fsw.memory = m['fsw_memory']
    else:
        print("  WARNING: checkpoint predates FSW - fsw results will be NaN")
    fair_gem.model.load_state_dict(m['fair_gem_model'])
    fair_gem.memory = m['fair_gem_memory']
    if 'fair_derpp_model' in m:
        fair_derpp.model.load_state_dict(m['fair_derpp_model'])
        fair_derpp.memory = m['fair_derpp_memory']
    else:
        print("  WARNING: checkpoint predates Fair-DER++ - fair_derpp results will be NaN")


def _assert_equal_memory_budget(**methods):
    """
    Every replay method must hold the same episodic-memory budget.

    Buffer size is the dominant factor in replay-based continual learning, so
    the budget is a constraint on all methods, not a per-method hyperparameter,
    and is checked here.
    """
    budgets = {name: (obj.memory.size_per_task, obj.mem_batch_size)
               for name, obj in methods.items()
               if getattr(obj, "memory", None) is not None}
    distinct = set(budgets.values())
    if len(distinct) > 1:
        detail = "\n".join(f"    {n:<12} size_per_task={s} mem_batch={b}"
                            for n, (s, b) in sorted(budgets.items()))
        raise SystemExit(
            "Replay methods have unequal memory budgets — the comparison would "
            f"not be valid:\n{detail}\n"
            "  Set every *_MEM_SIZE / *_MEM_BATCH in config.py to the same value.")
    if budgets:
        size, batch = next(iter(distinct))
        print(f"  memory budget: {size}/task, batch {batch} "
              f"(equal across {len(budgets)} replay methods)")


def _time_now():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _save_checkpoint(t_idx, acc, bal_acc, eo, dp, extra, runtime, proj_stats_log,
                     joint_retrain, finetune, agem, der_pp, group_dro, fairl, fsw, fair_gem, fair_derpp):
    torch.save({
        'last_completed_task': t_idx,
        'in_progress_task':    None,
        'in_progress_done':    [],
        'acc': acc, 'bal_acc': bal_acc, 'eo': eo, 'dp': dp, 'extra': extra,
        'runtime': runtime,
        'proj_stats_log': proj_stats_log,
        'state': _collect_state(joint_retrain, finetune, agem, der_pp,
                                group_dro, fairl, fsw, fair_gem, fair_derpp),
    }, CHECKPOINT_PATH)
    print(f"  [checkpoint saved (task {t_idx+1}/{N_TASKS})]")


def _save_progress(t_idx, last_completed, done_methods, acc, bal_acc, eo, dp, extra,
                   runtime, proj_stats_log,
                   joint_retrain, finetune, agem, der_pp, group_dro, fairl, fsw, fair_gem, fair_derpp):
    torch.save({
        'last_completed_task': last_completed,
        'in_progress_task':    t_idx,
        'in_progress_done':    done_methods,
        'acc': acc, 'bal_acc': bal_acc, 'eo': eo, 'dp': dp, 'extra': extra,
        'runtime': runtime,
        'proj_stats_log': proj_stats_log,
        'state': _collect_state(joint_retrain, finetune, agem, der_pp,
                                group_dro, fairl, fsw, fair_gem, fair_derpp),
    }, CHECKPOINT_PATH)


def _load_prev_results(methods_to_skip):
    """Load results from a previous partial run so we don't overwrite them."""
    if not methods_to_skip or not os.path.exists(RESULTS_PATH):
        return None
    try:
        prev     = np.load(RESULTS_PATH, allow_pickle=True)
        prev_acc = prev['acc'].item()
        if next(iter(prev_acc.values())).shape != (N_TASKS, N_TASKS):
            print(f"  WARNING: results.npz shape mismatch - skipping merge")
            return None
        group_prev = {}
        if os.path.exists(RESULTS_GROUPS_PATH):
            group_prev = np.load(RESULTS_GROUPS_PATH, allow_pickle=True).item()
        dp_prev = prev.get('dp', np.array({}, dtype=object)).item()
        if not isinstance(dp_prev, dict):
            dp_prev = {}
        runtime_prev = (prev['runtime'].item() if 'runtime' in prev.files else {})
        extra_prev = {k: prev[k].item() for k in EXTRA_METRICS if k in prev.files}
        return (prev_acc, prev['bal_acc'].item(), prev['eo'].item(), dp_prev,
                group_prev, runtime_prev, extra_prev)
    except Exception as e:
        print(f"  WARNING: results.npz merge failed ({e})")
        return None


def run():
    print("Loading data...")
    tasks, _ = get_spurious_tasks(seed=DATA_SEED)
    input_dim = tasks[0]["X_train"].shape[1]

    acc = {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS}
    bal_acc = {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS}
    eo = {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS}
    dp = {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS}
    extra = {k: {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS}
             for k in EXTRA_METRICS}
    runtime = {m: np.full(N_TASKS, np.nan) for m in METHODS}
    per_group_bal_acc = {m: {} for m in METHODS}
    proj_stats_log = {"fair_gem": {}, "fair_derpp": {}, "fsw": {}}

    joint_retrain = JointRetrain(input_dim, device=DEVICE)
    finetune = Finetune(input_dim, device=DEVICE)
    agem = AGEM(input_dim, device=DEVICE)
    der_pp = DERpp(input_dim, device=DEVICE)
    group_dro = GroupDRO(input_dim, device=DEVICE)
    fairl = FAIRL(input_dim, device=DEVICE)
    fsw = FSW(input_dim, device=DEVICE)
    fair_gem = FAIRGEM(input_dim, device=DEVICE, use_topk_dro=False, use_disparity=True, dynamic_eps=True)
    fair_derpp = FairDERpp(input_dim, device=DEVICE)

    _assert_equal_memory_budget(agem=agem, der_pp=der_pp, fairl=fairl, fsw=fsw,
                                fair_gem=fair_gem, fair_derpp=fair_derpp)

    methods_to_run = _parse_methods(sys.argv)
    methods_to_skip = set(METHODS) - methods_to_run
    novel_only = methods_to_run == set(NOVEL_METHODS)
    baselines_only = methods_to_run == set(BASELINE_METHODS)
    partial_fresh = "--fresh" in sys.argv and methods_to_run != set(METHODS)
    print(f"Methods: {sorted(methods_to_run)}")

    if novel_only:
        if not os.path.exists(BASELINE_CHECKPOINT_PATH):
            raise SystemExit(
                f"--novel-only requires {BASELINE_CHECKPOINT_PATH}. Run --baselines-only first.")
        bckpt = torch.load(BASELINE_CHECKPOINT_PATH, weights_only=False)
        for m in BASELINE_METHODS:
            if m in bckpt.get('acc', {}):
                acc[m] = bckpt['acc'][m]; bal_acc[m] = bckpt['bal_acc'][m]
                eo[m]  = bckpt['eo'][m];  dp[m]      = bckpt['dp'][m]
                if m in bckpt.get('runtime', {}):
                    runtime[m] = bckpt['runtime'][m]
        _restore_state(bckpt['state'], joint_retrain, finetune, agem, der_pp,
                       group_dro, fairl, fsw, fair_gem, fair_derpp)
        print(f"  Loaded baseline results from {BASELINE_CHECKPOINT_PATH}")

    start_task = 0
    skip_methods = set()
    last_completed = -1
    fresh = "--fresh" in sys.argv

    if not fresh and os.path.exists(CHECKPOINT_PATH):
        print(f"Found {CHECKPOINT_PATH} - resuming.")
        ckpt = torch.load(CHECKPOINT_PATH, weights_only=False)
        if next(iter(ckpt['acc'].values())).shape[0] != N_TASKS:
            raise SystemExit("Checkpoint task count doesn't match current config. Run with --fresh.")
        acc, bal_acc, eo = ckpt['acc'], ckpt['bal_acc'], ckpt['eo']
        dp = ckpt.get('dp', {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS})
        extra = ckpt.get('extra', extra)
        runtime = ckpt.get('runtime', runtime)
        proj_stats_log = ckpt.get('proj_stats_log', proj_stats_log)
        _restore_state(ckpt['state'], joint_retrain, finetune, agem, der_pp,
                       group_dro, fairl, fsw, fair_gem, fair_derpp)
        last_completed = ckpt['last_completed_task']
        if ckpt.get('in_progress_task') is not None:
            start_task   = ckpt['in_progress_task']
            skip_methods = set(ckpt.get('in_progress_done', []))
            print(f"  Resuming mid-task {start_task+1}/{N_TASKS}, already done: {sorted(skip_methods)}")
        else:
            start_task = last_completed + 1
            print(f"  Resuming from task {start_task+1}/{N_TASKS}")

    elif fresh and not partial_fresh and os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("--fresh: removed checkpoint, starting from scratch.")

    elif partial_fresh:
        if os.path.exists(CHECKPOINT_PATH):
            ckpt = torch.load(CHECKPOINT_PATH, weights_only=False)
            _restore_state(ckpt['state'], joint_retrain, finetune, agem, der_pp,
                           group_dro, fairl, fsw, fair_gem, fair_derpp)
            proj_stats_log = ckpt.get('proj_stats_log', proj_stats_log)
        # Re-create only the selected methods with fresh random weights
        if 'joint_retrain' in methods_to_run: joint_retrain = JointRetrain(input_dim, device=DEVICE)
        if 'finetune' in methods_to_run: finetune      = Finetune(input_dim, device=DEVICE)
        if 'agem' in methods_to_run: agem          = AGEM(input_dim, device=DEVICE)
        if 'der_pp' in methods_to_run: der_pp        = DERpp(input_dim, device=DEVICE)
        if 'group_dro' in methods_to_run: group_dro     = GroupDRO(input_dim, device=DEVICE)
        if 'fairl' in methods_to_run: fairl         = FAIRL(input_dim, device=DEVICE)
        if 'fsw' in methods_to_run: fsw           = FSW(input_dim, device=DEVICE)
        if 'fair_gem' in methods_to_run:
            fair_gem = FAIRGEM(input_dim, device=DEVICE, use_topk_dro=False, use_disparity=True, dynamic_eps=True)
        if 'fair_derpp' in methods_to_run: fair_derpp    = FairDERpp(input_dim, device=DEVICE)
        start_task = 0
        print(f"--fresh (partial): re-initialised {sorted(methods_to_run)}, kept others from checkpoint.")

    if novel_only:
        # Re-apply the baseline result matrices after the resume block, so a
        # resume from checkpoint.pt does not overwrite the
        # baseline_checkpoint.pt results loaded above.
        for m in BASELINE_METHODS:
            if m in bckpt.get('acc', {}):
                acc[m] = bckpt['acc'][m]; bal_acc[m] = bckpt['bal_acc'][m]
                eo[m]  = bckpt['eo'][m];  dp[m]      = bckpt['dp'][m]
                if m in bckpt.get('runtime', {}):
                    runtime[m] = bckpt['runtime'][m]

    method_obj_for = {
        "joint_retrain": joint_retrain, "finetune": finetune,
        "agem": agem, "der_pp": der_pp,
        "group_dro": group_dro, "fairl": fairl, "fsw": fsw,
        "fair_gem": fair_gem, "fair_derpp": fair_derpp,
    }

    # Carry over prior results for any methods we're not retraining this run
    if methods_to_skip:
        prev = _load_prev_results(methods_to_skip)
        if prev is not None:
            prev_acc, prev_bal, prev_eo, prev_dp, group_prev, runtime_prev, extra_prev = prev
            for m in methods_to_skip:
                if m in prev_acc: acc[m] = prev_acc[m]
                if m in prev_bal: bal_acc[m] = prev_bal[m]
                if m in prev_eo: eo[m] = prev_eo[m]
                if m in prev_dp: dp[m] = prev_dp[m]
                if m in group_prev: per_group_bal_acc[m] = group_prev[m]
                if m in runtime_prev: runtime[m] = runtime_prev[m]
                for k, d_prev in extra_prev.items():
                    if m in d_prev: extra[k][m] = d_prev[m]
            merged = sorted(m for m in methods_to_skip if m in prev_acc)
            if merged:
                print(f"  Merged previous results for: {merged}")

    for t_idx in range(start_task, N_TASKS):
        task = tasks[t_idx]
        state = task["state"]
        X_tr, y_tr, s_tr = task["X_train"], task["y_train"], task["s_train"]
        print(f"\n{'='*60}\nTask {t_idx+1}/{N_TASKS}: {state}  ({len(X_tr):,} samples)\n{'='*60}")

        resume_skip = skip_methods if t_idx == start_task else set()
        current_done = list(methods_to_skip | resume_skip)

        def _progress():
            _save_progress(t_idx, last_completed, current_done, acc, bal_acc, eo, dp, extra,
                           runtime, proj_stats_log,
                           joint_retrain, finetune, agem, der_pp,
                           group_dro, fairl, fsw, fair_gem, fair_derpp)

        def _train(name, fn):
            if name not in current_done:
                t0 = _time_now()
                fn()
                runtime[name][t_idx] = _time_now() - t0
                print(f"  {name:<14} {runtime[name][t_idx]:.1f}s")
                current_done.append(name)
                _progress()
            else:
                print(f"  {name:<14} (skipped)")

        _train("joint_retrain", lambda: joint_retrain.train_task(X_tr, y_tr))
        _train("finetune",  lambda: finetune.train_task(X_tr, y_tr))
        _train("agem", lambda: (agem.train_task(X_tr, y_tr, s_tr),   agem.remember(X_tr, y_tr, s_tr)))
        _train("der_pp", lambda: (der_pp.train_task(X_tr, y_tr, s_tr), der_pp.remember(X_tr, y_tr, s_tr)))
        _train("group_dro", lambda: group_dro.train_task(X_tr, y_tr, s_tr))
        _train("fairl", lambda: (fairl.train_task(X_tr, y_tr, s_tr),  fairl.remember(X_tr, y_tr, s_tr)))
        def _train_fsw():
            fsw.train_task(X_tr, y_tr, s_tr)
            fsw.remember(X_tr, y_tr, s_tr)
            ws = fsw.weight_stats.get(t_idx, [])
            if ws:
                proj_stats_log["fsw"][t_idx] = {"weight_epochs": ws}
                r = ws[-1]
                print(f"    weights: kept {r['kept']}/{r['total']}  "
                      f"mean_w s0={r['mean_w_s0']:.3f} s1={r['mean_w_s1']:.3f}")
        _train("fsw", _train_fsw)
        def _train_fair_gem():
            fair_gem.train_task(X_tr, y_tr, s_tr)
            fair_gem.remember(X_tr, y_tr, s_tr)
            ps = getattr(fair_gem, "proj_stats", None)
            if ps is not None:
                proj_stats_log["fair_gem"][t_idx] = dict(ps)
                print(f"    safety={ps.get('safety_proj_rate', float('nan')):.3f} "
                      f"mem={ps.get('mem_proj_rate', float('nan')):.3f} "
                      f"switch={ps.get('worst_group_switch_rate', float('nan')):.3f}")

        def _train_fair_derpp():
            fair_derpp.train_task(X_tr, y_tr, s_tr)
            fair_derpp.remember(X_tr, y_tr, s_tr)
            ps = getattr(fair_derpp, "proj_stats", None)
            if ps is not None:
                proj_stats_log["fair_derpp"][t_idx] = dict(ps)
                print(f"    safety={ps.get('safety_proj_rate', float('nan')):.3f} "
                      f"switch={ps.get('worst_group_switch_rate', float('nan')):.3f}")

        _train("fair_gem", _train_fair_gem)
        _train("fair_derpp", _train_fair_derpp)

        for e_idx in range(t_idx + 1):
            et = tasks[e_idx]
            yv, sv = et["y_test"], et["s_test"]

            def record(m, preds):
                acc[m][t_idx, e_idx] = task_accuracy(yv, preds)
                bal_acc[m][t_idx, e_idx] = balanced_accuracy_score(yv, preds)
                eo[m][t_idx, e_idx] = accuracy_parity_gap(yv, preds, sv)
                dp[m][t_idx, e_idx] = top_class_rate_gap(preds, sv)
                extra["eo_true"][m][t_idx, e_idx] = equalized_odds_gap(yv, preds, sv)
                # Same min_group_size as the gap metrics above (their default,
                # 100), so every group-level number in this row is computed over
                # the same group set.
                pg = per_group_accuracy(yv, preds, sv, min_group_size=100)
                extra["worst_group"][m][t_idx, e_idx] = min(pg.values()) if pg else np.nan
                for g, g_acc in pg.items():
                    if g not in per_group_bal_acc[m]:
                        per_group_bal_acc[m][g] = np.full((N_TASKS, N_TASKS), np.nan)
                    per_group_bal_acc[m][g][t_idx, e_idx] = g_acc

            for m in METHODS:
                if m in methods_to_run:
                    record(m, method_obj_for[m].predict(et["X_test"]))
                    # Surrogate-validation matrices: eval-time CE group-loss
                    # disparities for every method, so surrogate/metric
                    # alignment is computable across methods x seeds x cells.
                    _losses = per_sample_ce(method_obj_for[m].model, et["X_test"], yv, DEVICE)
                    extra["loss_gap"][m][t_idx, e_idx] = group_loss_disparity(_losses, sv)
                    extra["loss_gap_cond"][m][t_idx, e_idx] = conditional_group_loss_disparity(_losses, yv, sv)

        np.save(PROJ_STATS_PATH, proj_stats_log, allow_pickle=True)
        _save_checkpoint(t_idx, acc, bal_acc, eo, dp, extra, runtime, proj_stats_log,
                         joint_retrain, finetune, agem, der_pp, group_dro, fairl, fsw, fair_gem, fair_derpp)
        if baselines_only:
            import shutil
            shutil.copy2(CHECKPOINT_PATH, BASELINE_CHECKPOINT_PATH)
            print(f"  [baseline checkpoint saved]")
        last_completed = t_idx
        skip_methods = set()

    task_names = [t["state"] for t in tasks]
    np.savez(RESULTS_PATH, acc=acc, bal_acc=bal_acc, eo=eo, dp=dp,
             runtime=runtime, methods=METHODS, states=task_names, seed=EFFECTIVE_SEED,
             **{k: extra[k] for k in EXTRA_METRICS})
    np.save(RESULTS_GROUPS_PATH, per_group_bal_acc, allow_pickle=True)
    np.save(PROJ_STATS_PATH, proj_stats_log, allow_pickle=True)
    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"\n{'-'*105}")
    print(f"{'Method':30s}  {'BalAcc':>8}  {'AccParity':>9}  {'TopRate':>8}  {'BWT':>8}  {'F-BWT':>8}")
    print(f"{'-'*105}")
    for m in METHODS:
        summarize(acc[m], bal_acc[m], eo[m], dp[m], m,
                  extra['eo_true'][m], extra['worst_group'][m])
    return acc, bal_acc, eo, dp


def reeval():
    """Write results_test.npz as a final-row VIEW of results.npz.

    The final row of each matrix (metrics of the final model on every task's
    test split) is kept and the rest set to NaN. Downstream plots read the
    final-model results by this name; it contains no information beyond
    results.npz.
    """
    if not os.path.exists(RESULTS_PATH):
        print(f"No results.npz under {OUT or '.'} - run train.py first.")
        return
    src = np.load(RESULTS_PATH, allow_pickle=True)
    skip = {"methods", "states", "seed", "runtime"}
    view = {}
    for k in src.files:
        if k in skip:
            continue
        out = {}
        for m, M in src[k].item().items():
            M = np.asarray(M, dtype=float)
            V = np.full_like(M, np.nan)
            V[-1, :] = M[-1, :]
            out[m] = V
        view[k] = out
    np.savez(RESULTS_TEST_PATH, methods=src["methods"], states=src["states"], **view)
    groups = np.load(RESULTS_GROUPS_PATH, allow_pickle=True).item()
    gview = {}
    for m, per_g in groups.items():
        gview[m] = {}
        for g, M in per_g.items():
            M = np.asarray(M, dtype=float)
            V = np.full_like(M, np.nan)
            V[-1, :] = M[-1, :]
            gview[m][g] = V
    np.save(RESULTS_TEST_GROUPS_PATH, gview, allow_pickle=True)
    print(f"Wrote final-row view: {RESULTS_TEST_PATH}")



if __name__ == "__main__":
    print(f"Run context: {runctx.describe(SEED_ARG)}")
    if "--reeval" in sys.argv:
        reeval()
    else:
        run()
