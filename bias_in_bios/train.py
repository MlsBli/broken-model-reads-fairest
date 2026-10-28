"""
train.py — Run all baselines sequentially across tasks and save results.

Usage (run from bias_in_bios/results; add --seed N for a seeded run):
    python ../train.py                  # train all methods (auto-resumes from checkpoint)
    python ../train.py --fresh          # start from scratch
    python ../train.py --reeval         # write the final-model test results
    python ../train.py --baselines-only # train baseline methods only
    python ../train.py --novel-only     # train fair_gem/fair_derpp only (loads baseline checkpoint)

Output: results.npz (training), results_test.npz (--reeval)
"""

import os
import sys
import numpy as np
import torch
import time
from sklearn.metrics import balanced_accuracy_score
from metrics import (group_loss_disparity, conditional_group_loss_disparity,
                     task_accuracy, accuracy_parity_gap, demographic_parity_gap,
                     equalized_odds_gap, worst_group_balanced_accuracy,
                     summarize,
                     SUMMARY_HEADER, per_group_accuracy, tpr_gap_rms)
from baselines_neural import JointRetrain, Finetune, AGEM, DERpp
from baselines_fairness import GroupDRO
from baselines_fairl import FaIRL
from fairgem import FAIRGEM
from fairderpp import FairDERpp
from fsw import FSW
from model import predict_and_ce
import factory
import config
import runctx
from config import DEVICE, DATASET, SPURIOUS_N_TASKS, BIOS_STEP_SIZE, _N_CLASSES_BY_DATASET, MLP_EPOCHS, N_CLASSES

# --seed N seeds the RNGs, varies the train/validation split, and redirects every
# artifact to runs/seedN/ so a sweep never overwrites the single-seed results.
# Without --seed, cfg.SEED is used and artifacts are written to ./.
# The test split is not resampled — see runctx for why.
SEED_ARG       = runctx.parse_seed(sys.argv)
EFFECTIVE_SEED = runctx.apply_seed(SEED_ARG)
DATA_SEED      = runctx.data_seed(SEED_ARG)
OUT            = runctx.output_dir(SEED_ARG)

METHODS = [
    "joint_retrain",
    "finetune", "agem", "derpp",
    "group_dro",
    "fairl",
    "fsw",
    "fair_gem",
    "fair_derpp",
]
BASELINE_METHODS = [
    "joint_retrain", "finetune", "agem", "derpp",
    "group_dro",
    "fairl",
    "fsw",
]
NOVEL_METHODS = ["fair_gem", "fair_derpp"]

if DATASET == "spurious_shift":
    N_TASKS = SPURIOUS_N_TASKS
elif DATASET == "bios":
    N_TASKS = (_N_CLASSES_BY_DATASET["bios"] + BIOS_STEP_SIZE - 1) // BIOS_STEP_SIZE
else:
    raise ValueError(f"Unknown DATASET={DATASET!r}")
CHECKPOINT_PATH          = os.path.join(OUT, "checkpoint.pt")
BASELINE_CHECKPOINT_PATH = os.path.join(OUT, "baseline_checkpoint.pt")


# Checkpoint helpers
def _collect_state(joint_retrain, finetune, agem, derpp, group_dro, fairl, fsw, fair_gem, fair_derpp):
    state = {
        'joint_retrain_model':     joint_retrain.model.state_dict() if joint_retrain else None,
        'finetune_model':          finetune.model.state_dict() if finetune else None,
        'agem_model':              agem.model.state_dict() if agem else None,
        'agem_memory':             agem.memory if agem else None,
        'derpp_model':             derpp.model.state_dict() if derpp else None,
        'derpp_memory':            derpp.memory if derpp else None,
        'group_dro_model':         group_dro.model.state_dict() if group_dro else None,
        'fsw_model':               fsw.model.state_dict() if fsw else None,
        'fsw_memory':              fsw.memory if fsw else None,
        'fair_gem_model':          fair_gem.model.state_dict() if fair_gem else None,
        'fair_gem_memory':         fair_gem.memory if fair_gem else None,
        'fair_derpp_model':        fair_derpp.model.state_dict() if fair_derpp else None,
        'fair_derpp_memory':       fair_derpp.memory if fair_derpp else None,
    }
    if fairl is not None:
        state['fairl_model']     = fairl.model.state_dict()
        state['fairl_exemplars'] = fairl._exemplars
    if joint_retrain is not None and hasattr(joint_retrain, 'all_X'):
        state['joint_retrain_data'] = (joint_retrain.all_X, joint_retrain.all_y)
    return state


def _restore_state(m, joint_retrain, finetune, agem, derpp, group_dro, fairl, fsw, fair_gem, fair_derpp):
    if joint_retrain is not None and m.get('joint_retrain_model') is not None:
        joint_retrain.model.load_state_dict(m['joint_retrain_model'])
        if 'joint_retrain_data' in m and hasattr(joint_retrain, 'all_X'):
            saved = m['joint_retrain_data']
            joint_retrain.all_X = saved[0]
            joint_retrain.all_y = saved[1]
    if finetune is not None and m.get('finetune_model') is not None:
        finetune.model.load_state_dict(m['finetune_model'])
    if agem is not None and m.get('agem_model') is not None:
        agem.model.load_state_dict(m['agem_model'])
        agem.memory = m['agem_memory']
    if derpp is not None:
        if 'derpp_model' in m and m['derpp_model'] is not None:
            derpp.model.load_state_dict(m['derpp_model'])
            derpp.memory = m['derpp_memory']
        elif 'derpp_model' not in m:
            print("  WARNING: checkpoint predates DER++ - derpp results will be NaN")
    if group_dro is not None:
        if 'group_dro_model' in m and m['group_dro_model'] is not None:
            group_dro.model.load_state_dict(m['group_dro_model'])
        elif 'group_dro_model' not in m:
            print("  WARNING: checkpoint predates GroupDRO state saving - group_dro results will be NaN")
    if fairl is not None and 'fairl_model' in m:
        fairl.model.load_state_dict(m['fairl_model'])
        fairl._exemplars = m.get('fairl_exemplars', [])
    if fsw is not None:
        if 'fsw_model' in m and m['fsw_model'] is not None:
            fsw.model.load_state_dict(m['fsw_model'])
            fsw.memory = m['fsw_memory']
        elif 'fsw_model' not in m:
            print("  WARNING: checkpoint predates FSW - fsw results will be NaN")
    if fair_gem is not None and m.get('fair_gem_model') is not None:
        fair_gem.model.load_state_dict(m['fair_gem_model'])
        fair_gem.memory = m['fair_gem_memory']
    if fair_derpp is not None:
        if 'fair_derpp_model' in m and m['fair_derpp_model'] is not None:
            fair_derpp.model.load_state_dict(m['fair_derpp_model'])
            fair_derpp.memory = m['fair_derpp_memory']
        elif 'fair_derpp_model' not in m:
            print("  WARNING: checkpoint predates Fair-DER++ - fair_derpp results will be NaN")


def _time_now():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _save_checkpoint(t_idx, acc, bal_acc, eo, dp, tpr, acc_parity, worst_group,
                     loss_gap, loss_gap_cond, runtime, proj_stats_log,
                     joint_retrain, finetune, agem, derpp, group_dro, fairl, fsw, fair_gem, fair_derpp):
    torch.save({
        'last_completed_task': t_idx,
        'in_progress_task':    None,
        'in_progress_done':    [],
        'dataset': config.DATASET,
        'acc':       acc,
        'bal_acc':   bal_acc,
        'eo':        eo,
        'dp':        dp,
        'tpr':       tpr,
        'acc_parity': acc_parity,
        'worst_group': worst_group,
        'loss_gap': loss_gap,
        'loss_gap_cond': loss_gap_cond,
        'runtime': runtime,
        'proj_stats_log': proj_stats_log,
        'state':   _collect_state(joint_retrain, finetune, agem, derpp,
                                  group_dro, fairl, fsw, fair_gem, fair_derpp),
    }, CHECKPOINT_PATH)
    print(f"  [checkpoint -> {CHECKPOINT_PATH}  (task {t_idx+1}/{N_TASKS} complete)]")


def _save_progress(t_idx, last_completed, done_methods, acc, bal_acc, eo, dp, tpr, acc_parity, worst_group,
                   loss_gap, loss_gap_cond, runtime, proj_stats_log,
                   joint_retrain, finetune, agem, derpp, group_dro, fairl, fsw, fair_gem, fair_derpp):
    torch.save({
        'last_completed_task': last_completed,
        'in_progress_task':    t_idx,
        'in_progress_done':    done_methods,
        'dataset': config.DATASET,
        'acc':       acc,
        'bal_acc':   bal_acc,
        'eo':        eo,
        'dp':        dp,
        'tpr':       tpr,
        'acc_parity': acc_parity,
        'worst_group': worst_group,
        'loss_gap': loss_gap,
        'loss_gap_cond': loss_gap_cond,
        'runtime': runtime,
        'proj_stats_log': proj_stats_log,
        'state':   _collect_state(joint_retrain, finetune, agem, derpp,
                                  group_dro, fairl, fsw, fair_gem, fair_derpp),
    }, CHECKPOINT_PATH)


# Main
def run():
    print("Loading data...")
    tasks, scaler = factory.get_tasks(seed=DATA_SEED)
    _bert_mode = factory.is_bert_task(tasks[0])
    input_dim  = tasks[0]["X_train"].shape[1]

    acc         = {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS}
    bal_acc     = {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS}
    eo          = {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS}
    dp          = {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS}
    tpr         = {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS}
    acc_parity  = {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS}
    worst_group = {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS}
    loss_gap      = {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS}
    loss_gap_cond = {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS}
    runtime     = {m: np.full(N_TASKS, np.nan) for m in METHODS}
    per_group_bal_acc = {m: {} for m in METHODS}
    proj_stats_log = {"fair_gem": {}, "fair_derpp": {}, "fsw": {}}

    baselines_only = "--baselines-only" in sys.argv
    novel_only     = "--novel-only"     in sys.argv
    if baselines_only and novel_only:
        raise SystemExit("--baselines-only and --novel-only are mutually exclusive.")

    if not novel_only:
        joint_retrain = JointRetrain(input_dim, device=DEVICE, bert_mode=_bert_mode,
                                     bert_lr=config.JOINT_BERT_LR)
        finetune      = Finetune(input_dim, device=DEVICE, bert_mode=_bert_mode,
                                 bert_lr=config.FINETUNE_BERT_LR)
        agem          = AGEM(input_dim, device=DEVICE, bert_mode=_bert_mode, bert_lr=config.AGEM_BERT_LR)
        derpp         = DERpp(input_dim, device=DEVICE, bert_mode=_bert_mode, bert_lr=config.DERPP_BERT_LR)
        group_dro     = GroupDRO(input_dim, device=DEVICE, bert_mode=_bert_mode, bert_lr=config.GROUPDRO_BERT_LR)
        fairl         = FaIRL(input_dim, device=DEVICE, bert_mode=_bert_mode, bert_lr=config.FAIRL_BERT_LR)
        fsw           = FSW(input_dim, device=DEVICE, bert_mode=_bert_mode, bert_lr=config.FSW_BERT_LR)
    else:
        joint_retrain = finetune = agem = derpp = group_dro = fairl = fsw = None
    if not baselines_only:
        fair_gem   = FAIRGEM(input_dim, device=DEVICE, use_topk_dro=False, use_disparity=True,
                             dynamic_eps=True, bert_mode=_bert_mode, bert_lr=config.FAIRGEM_BERT_LR)
        fair_derpp = FairDERpp(input_dim, device=DEVICE, bert_mode=_bert_mode, bert_lr=config.FAIRDERPP_BERT_LR)
    else:
        fair_gem = fair_derpp = None

    if novel_only:
        if not os.path.exists(BASELINE_CHECKPOINT_PATH):
            raise SystemExit(
                f"--novel-only requires {BASELINE_CHECKPOINT_PATH}.\n"
                "Run --baselines-only first to generate it."
            )
        bckpt = torch.load(BASELINE_CHECKPOINT_PATH, weights_only=False)
        for m in BASELINE_METHODS:
            if m in bckpt.get('acc', {}):
                acc[m]     = bckpt['acc'][m]
                bal_acc[m] = bckpt['bal_acc'][m]
                eo[m]      = bckpt['eo'][m]
                dp[m]      = bckpt['dp'][m]
                if 'tpr' in bckpt and m in bckpt['tpr']:
                    tpr[m] = bckpt['tpr'][m]
                if 'acc_parity' in bckpt and m in bckpt['acc_parity']:
                    acc_parity[m] = bckpt['acc_parity'][m]
                if 'worst_group' in bckpt and m in bckpt['worst_group']:
                    worst_group[m] = bckpt['worst_group'][m]
                if 'loss_gap' in bckpt and m in bckpt['loss_gap']:
                    loss_gap[m] = bckpt['loss_gap'][m]
                if 'loss_gap_cond' in bckpt and m in bckpt['loss_gap_cond']:
                    loss_gap_cond[m] = bckpt['loss_gap_cond'][m]
                if m in bckpt.get('runtime', {}):
                    runtime[m] = bckpt['runtime'][m]
        _restore_state(bckpt['state'], joint_retrain,
                       finetune, agem, derpp, group_dro, fairl, fsw, fair_gem, fair_derpp)
        proj_stats_log = bckpt.get('proj_stats_log', proj_stats_log)
        print(f"  Loaded baseline results from {BASELINE_CHECKPOINT_PATH} "
              f"({bckpt['last_completed_task']+1}/{N_TASKS} tasks complete)")

    start_task     = 0
    skip_methods   = set()
    last_completed = -1
    fresh = "--fresh" in sys.argv
    if not fresh and os.path.exists(CHECKPOINT_PATH):
        print(f"Found {CHECKPOINT_PATH} - resuming. (run with --fresh to start over)")
        print("  WARNING: checkpoints hold model weights and memory buffers only - "
              "AdamW moments, LR-scheduler position, and FaIRL's discriminator "
              "restart at the resume point, so a resumed run is not bit-identical "
              "to an uninterrupted one. Prefer --fresh for publication runs.")
        ckpt = torch.load(CHECKPOINT_PATH, weights_only=False)
        ckpt_dataset = ckpt.get('dataset')
        if ckpt_dataset is not None and ckpt_dataset != config.DATASET:
            raise SystemExit(
                f"Checkpoint was trained on DATASET={ckpt_dataset!r} but config.DATASET={config.DATASET!r}. "
                f"Run with --fresh or restore the matching config to resume."
            )
        ckpt_n = next(iter(ckpt['acc'].values())).shape[0]
        if ckpt_n != N_TASKS:
            raise SystemExit(
                f"Checkpoint has {ckpt_n} tasks but current config has N_TASKS={N_TASKS}. "
                "Run with --fresh or restore the matching DATASET config."
            )
        acc, bal_acc, eo = ckpt['acc'], ckpt['bal_acc'], ckpt['eo']
        dp          = ckpt.get('dp',          {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS})
        tpr         = ckpt.get('tpr',         {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS})
        acc_parity  = ckpt.get('acc_parity',  {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS})
        worst_group = ckpt.get('worst_group', {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS})
        loss_gap      = ckpt.get('loss_gap',      {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS})
        loss_gap_cond = ckpt.get('loss_gap_cond', {m: np.full((N_TASKS, N_TASKS), np.nan) for m in METHODS})
        runtime     = ckpt.get('runtime', {m: np.full(N_TASKS, np.nan) for m in METHODS})
        _restore_state(ckpt['state'], joint_retrain, finetune, agem, derpp,
                       group_dro, fairl, fsw, fair_gem, fair_derpp)
        proj_stats_log = ckpt.get('proj_stats_log', proj_stats_log)
        last_completed = ckpt['last_completed_task']
        def _task_name(idx):
            return tasks[idx].get("state", f"task_{idx}")
        if ckpt.get('in_progress_task') is not None:
            start_task   = ckpt['in_progress_task']
            skip_methods = set(ckpt.get('in_progress_done', []))
            print(f"  Resuming mid-task {start_task+1}/{N_TASKS} ({_task_name(start_task)}), "
                  f"skipping: {sorted(skip_methods)}")
        else:
            start_task = last_completed + 1
            print(f"  Resuming from task {start_task+1}/{N_TASKS} ({_task_name(start_task)})")
    elif fresh and os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("--fresh: removed existing checkpoint, starting from scratch.")

    def _seen_classes_after(t_idx):
        if DATASET != "bios":
            return None
        hi = min(N_CLASSES, (t_idx + 1) * BIOS_STEP_SIZE)
        return list(range(hi))

    for t_idx in range(start_task, N_TASKS):
        task    = tasks[t_idx]
        state   = task.get("state", f"task_{t_idx}")
        n_samp  = len(task["X_train"])
        print(f"\n{'='*60}\nTask {t_idx+1}/{N_TASKS}: {state}  ({n_samp:,} samples)\n{'='*60}")

        seen         = _seen_classes_after(t_idx)
        _task_epochs = config.BIOS_BERT_EPOCHS if _bert_mode else MLP_EPOCHS
        current_done = list(skip_methods) if t_idx == start_task else []
        if baselines_only:
            current_done = list(set(current_done) | set(NOVEL_METHODS))
        if novel_only:
            current_done = list(set(current_done) | set(BASELINE_METHODS))

        def _progress():
            # ~45 full-state saves per seed (9 methods x 5 tasks), each holding
            # nine BERT-base state_dicts -- ~200 GB of writes per seed, which on
            # a network volume costs more wall-clock than the training does.
            # They only buy mid-task crash resume; CLFAIR_NO_PROGRESS_SAVE=1
            # keeps the end-of-task checkpoints and drops these. Results are
            # unaffected either way -- nothing reads them but a resumed run.
            if os.environ.get("CLFAIR_NO_PROGRESS_SAVE") == "1":
                return
            _save_progress(t_idx, last_completed, current_done, acc, bal_acc, eo, dp, tpr,
                           acc_parity, worst_group, loss_gap, loss_gap_cond, runtime, proj_stats_log,
                           joint_retrain, finetune, agem, derpp, group_dro, fairl, fsw, fair_gem, fair_derpp)

        X_tr       = task["token_ids_train"] if _bert_mode else task["X_train"]
        y_tr, s_tr = task["y_train"], task["s_train"]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if "joint_retrain" not in current_done:
            t0 = _time_now()
            joint_retrain.train_task(X_tr, y_tr, epochs=_task_epochs, seen_classes=seen)
            runtime["joint_retrain"][t_idx] = _time_now() - t0
            print(f"  joint_retrain {runtime['joint_retrain'][t_idx]:.1f}s")
            current_done.append("joint_retrain"); _progress()
        else:
            print("  joint_retrain (skipped)")

        if "finetune" not in current_done:
            t0 = _time_now()
            finetune.train_task(X_tr, y_tr, epochs=_task_epochs, seen_classes=seen)
            runtime["finetune"][t_idx] = _time_now() - t0
            print(f"  finetune      {runtime['finetune'][t_idx]:.1f}s")
            current_done.append("finetune"); _progress()
        else:
            print("  finetune      (skipped)")

        if "agem" not in current_done:
            t0 = _time_now()
            agem.train_task(X_tr, y_tr, s_tr, epochs=_task_epochs, seen_classes=seen)
            agem.remember(X_tr, y_tr, s_tr)
            runtime["agem"][t_idx] = _time_now() - t0
            print(f"  agem          {runtime['agem'][t_idx]:.1f}s")
            current_done.append("agem"); _progress()
        else:
            print("  agem          (skipped)")

        if "derpp" not in current_done:
            t0 = _time_now()
            derpp.train_task(X_tr, y_tr, s_tr, epochs=_task_epochs, seen_classes=seen)
            derpp.remember(X_tr, y_tr, s_tr)
            runtime["derpp"][t_idx] = _time_now() - t0
            print(f"  derpp         {runtime['derpp'][t_idx]:.1f}s")
            current_done.append("derpp"); _progress()
        else:
            print("  derpp         (skipped)")

        if "group_dro" not in current_done:
            t0 = _time_now()
            group_dro.train_task(X_tr, y_tr, s_tr, epochs=_task_epochs, seen_classes=seen)
            runtime["group_dro"][t_idx] = _time_now() - t0
            print(f"  group_dro     {runtime['group_dro'][t_idx]:.1f}s")
            current_done.append("group_dro"); _progress()
        else:
            print("  group_dro     (skipped)")

        if "fairl" not in current_done:
            t0 = _time_now()
            fairl.train_task(X_tr, y_tr, s_tr, epochs=_task_epochs, seen_classes=seen)
            fairl.remember(X_tr, y_tr, s_tr)
            runtime["fairl"][t_idx] = _time_now() - t0
            print(f"  fairl         {runtime['fairl'][t_idx]:.1f}s")
            current_done.append("fairl"); _progress()
        else:
            print("  fairl         (skipped)")

        if "fsw" not in current_done:
            t0 = _time_now()
            fsw.train_task(X_tr, y_tr, s_tr, epochs=_task_epochs, seen_classes=seen)
            fsw.remember(X_tr, y_tr, s_tr)
            runtime["fsw"][t_idx] = _time_now() - t0
            ws = fsw.weight_stats.get(fsw.memory.num_tasks() - 1, [])
            if ws:
                proj_stats_log["fsw"][t_idx] = {"weight_epochs": ws}
                r = ws[-1]
                print(f"  fsw           {runtime['fsw'][t_idx]:.1f}s  "
                      f"[kept {r['kept']}/{r['total']} "
                      f"w_s0={r['mean_w_s0']:.3f} w_s1={r['mean_w_s1']:.3f}]")
            else:
                print(f"  fsw           {runtime['fsw'][t_idx]:.1f}s")
            current_done.append("fsw"); _progress()
        else:
            print("  fsw           (skipped)")

        if "fair_gem" not in current_done:
            t0 = _time_now()
            fair_gem.train_task(X_tr, y_tr, s_tr, epochs=_task_epochs, seen_classes=seen)
            fair_gem.remember(X_tr, y_tr, s_tr)
            runtime["fair_gem"][t_idx] = _time_now() - t0
            ps = getattr(fair_gem, "proj_stats", None)
            if ps is not None:
                proj_stats_log["fair_gem"][t_idx] = dict(ps)
                print(f"  fair_gem      {runtime['fair_gem'][t_idx]:.1f}s  "
                      f"[safety_proj_rate={ps.get('safety_proj_rate', float('nan')):.3f} "
                      f"mem_proj_rate={ps.get('mem_proj_rate', float('nan')):.3f}]")
            else:
                print(f"  fair_gem      {runtime['fair_gem'][t_idx]:.1f}s")
            current_done.append("fair_gem"); _progress()
        else:
            print("  fair_gem      (skipped)")

        if "fair_derpp" not in current_done:
            t0 = _time_now()
            fair_derpp.train_task(X_tr, y_tr, s_tr, epochs=_task_epochs, seen_classes=seen)
            fair_derpp.remember(X_tr, y_tr, s_tr)
            runtime["fair_derpp"][t_idx] = _time_now() - t0
            ps = getattr(fair_derpp, "proj_stats", None)
            if ps is not None:
                proj_stats_log["fair_derpp"][t_idx] = dict(ps)
                print(f"  fair_derpp    {runtime['fair_derpp'][t_idx]:.1f}s  "
                      f"[safety_proj_rate={ps.get('safety_proj_rate', float('nan')):.3f}]")
            else:
                print(f"  fair_derpp    {runtime['fair_derpp'][t_idx]:.1f}s")
            current_done.append("fair_derpp"); _progress()
        else:
            print("  fair_derpp    (skipped)")

        for e_idx in range(t_idx + 1):
            et       = tasks[e_idx]
            Xv       = et["token_ids_test"] if _bert_mode else et["X_test"]
            yv, sv   = et["y_test"], et["s_test"]

            def record(m, preds):
                acc[m][t_idx, e_idx]        = task_accuracy(yv, preds)
                bal_acc[m][t_idx, e_idx]    = balanced_accuracy_score(yv, preds)
                eo[m][t_idx, e_idx]         = equalized_odds_gap(yv, preds, sv)
                dp[m][t_idx, e_idx]         = demographic_parity_gap(preds, sv)
                tpr[m][t_idx, e_idx]        = tpr_gap_rms(yv, preds, sv)
                acc_parity[m][t_idx, e_idx] = accuracy_parity_gap(yv, preds, sv)
                worst_group[m][t_idx, e_idx]= worst_group_balanced_accuracy(yv, preds, sv)
                for g, g_acc in per_group_accuracy(yv, preds, sv, min_group_size=20).items():
                    if g not in per_group_bal_acc[m]:
                        per_group_bal_acc[m][g] = np.full((N_TASKS, N_TASKS), np.nan)
                    per_group_bal_acc[m][g][t_idx, e_idx] = g_acc

            # One eval forward per method per cell: the argmax feeds the
            # fairness metrics (identical to .predict()), the per-sample CE
            # feeds the surrogate-validation matrices.
            _eval_objs = ([] if novel_only else [
                ("joint_retrain", joint_retrain), ("finetune", finetune),
                ("agem", agem), ("group_dro", group_dro), ("fairl", fairl),
                ("fsw", fsw), ("derpp", derpp),
            ]) + ([] if baselines_only else [
                ("fair_gem", fair_gem), ("fair_derpp", fair_derpp),
            ])
            for _m, _obj in _eval_objs:
                _preds, _losses = predict_and_ce(_obj.model, Xv, yv, DEVICE, seen_classes=seen)
                record(_m, _preds)
                loss_gap[_m][t_idx, e_idx] = group_loss_disparity(_losses, sv)
                loss_gap_cond[_m][t_idx, e_idx] = conditional_group_loss_disparity(_losses, yv, sv)

        _save_checkpoint(t_idx, acc, bal_acc, eo, dp, tpr, acc_parity, worst_group,
                         loss_gap, loss_gap_cond, runtime, proj_stats_log,
                         joint_retrain, finetune, agem, derpp, group_dro, fairl, fsw, fair_gem, fair_derpp)
        if baselines_only:
            import shutil
            shutil.copy2(CHECKPOINT_PATH, BASELINE_CHECKPOINT_PATH)
            print(f"  [baseline checkpoint -> {BASELINE_CHECKPOINT_PATH}]")
        last_completed = t_idx
        skip_methods   = set()

    task_names = [t.get("state", f"task_{i}") for i, t in enumerate(tasks)]
    np.savez(os.path.join(OUT, "results.npz"), acc=acc, bal_acc=bal_acc, eo=eo, dp=dp, tpr=tpr,
             acc_parity=acc_parity, worst_group=worst_group,
             loss_gap=loss_gap, loss_gap_cond=loss_gap_cond, runtime=runtime,
             methods=METHODS, states=task_names)
    np.save(os.path.join(OUT, "results_groups.npy"), per_group_bal_acc, allow_pickle=True)
    np.save(os.path.join(OUT, "results_proj_stats.npy"), proj_stats_log, allow_pickle=True)
    print(f"\nResults saved to results.npz + results_groups.npy")
    print(f"\n{'-'*100}")
    print(SUMMARY_HEADER)
    print(f"{'-'*100}")
    for m in METHODS:
        summarize(acc[m], bal_acc[m], worst_group[m], tpr[m], eo[m], dp[m], acc_parity[m], m)
    return acc, bal_acc, eo, dp, tpr, acc_parity, worst_group


def reeval():
    """Write results_test.npz as a final-row VIEW of results.npz.

    run() already evaluates every seen task on the test split, so this
    recomputes nothing: it extracts the final-model results and writes them
    under the name downstream plots read.
    """
    if not os.path.exists(os.path.join(OUT, "results.npz")):
        print(f"No results.npz under {OUT or '.'} - run train.py first.")
        return
    src = np.load(os.path.join(OUT, "results.npz"), allow_pickle=True)
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
    np.savez(os.path.join(OUT, "results_test.npz"), methods=src["methods"], states=src["states"], **view)
    groups = np.load(os.path.join(OUT, "results_groups.npy"), allow_pickle=True).item()
    gview = {}
    for m, per_g in groups.items():
        gview[m] = {}
        for g, M in per_g.items():
            M = np.asarray(M, dtype=float)
            V = np.full_like(M, np.nan)
            V[-1, :] = M[-1, :]
            gview[m][g] = V
    np.save(os.path.join(OUT, "results_test_groups.npy"), gview, allow_pickle=True)
    print("Wrote final-row view: " + os.path.join(OUT, "results_test.npz"))



if __name__ == "__main__":
    if os.environ.get("CLFAIR_SKIP_PRETRAINED") == "1" and "--reeval" not in sys.argv:
        raise SystemExit(
            "CLFAIR_SKIP_PRETRAINED=1 is set. That builds BERT from config with "
            "random weights, which is correct only when a checkpoint is restored "
            "immediately afterwards. Training from it would silently discard the "
            "pretrained initialisation. Unset it and rerun.")
    print(f"Run context: {runctx.describe(SEED_ARG)}")
    if "--reeval" in sys.argv:
        reeval()
    else:
        run()
