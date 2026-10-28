"""
config.py — Hyperparameters for all methods.

All neural baselines share the same architecture and optimizer; only the
training objective differs between methods.
"""

DATASET = "bios"        # "spurious_shift" | "bios"

_N_CLASSES_BY_DATASET = {
    "spurious_shift": 2,
    # 25 = the most frequent professions; the 3 rarest formed an
    # unlearnable 3-class sixth task (see datasets/bios_tasks.py).
    "bios":           25,
}
N_CLASSES = _N_CLASSES_BY_DATASET.get(DATASET, 2)

# Disparity-direction smoothing for FAIR-GEM / Fair-DER++ (0.0 = off, the
# default). A value such as 0.9 ranks worst/best group by an EMA of per-group
# losses (~10-step memory) instead of the noisy single batch.
SAFETY_EMA = 0.0

# Architecture (shared across all neural methods)
MLP_HIDDEN_DIM    = 256
MLP_DROPOUT       = 0.1
MLP_LR            = 3e-4
MLP_WEIGHT_DECAY  = 1e-4
MLP_EPOCHS        = 30
MLP_BATCH_SIZE    = 512
MLP_WARMUP_EPOCHS = 3
MLP_GRAD_CLIP     = 3.0
MLP_PATIENCE      = 5
MLP_MIN_DELTA     = 1e-3

# Memory (A-GEM, DER++, FAIR-GEM, Fair-DER++; BIOS FaIRL-insp. uses none)
MEMORY_SIZE_PER_TASK = 800
MEMORY_STRATIFIED    = True
AGEM_MEM_BATCH_SIZE  = 448

# GroupDRO
DRO_ETA = 0.021170810121534184
DRO_ADJ = 2.0997267082736646

# DER++
DERPP_ALPHA = 0.30710573677773717
DERPP_BETA  = 1.7254716573280358

# FAIR-GEM
FAIRGEM_TOP_K = 64
# eps_t = min(alpha, beta * disparity_loss), where g_safety = direction of increasing disparity.
# g_safety is L2-normalized, so eps is scale-invariant w.r.t. gradient magnitude.
FAIRGEM_ALPHA = 0.24437525838895208
FAIRGEM_BETA  = 1.8105311308911887

# Fair-DER++
FAIRDERPP_ALPHA     = 0.2801444283492551
FAIRDERPP_BETA      = 1.9242870058652546
FAIRDERPP_EPS_ALPHA = 0.37920959915707925
FAIRDERPP_EPS_BETA  = 1.0568737096007825

# FSW (Park et al. 2026) — EO variant; ranges follow the reference Bios_EO.sh grid
FSW_ALPHA          = 0.000524833536559974   # gradient-adjustment scale in the linearized group loss
FSW_LAMBDA         = 0.15314724285373518     # current-class accuracy regularizer
FSW_LAMBDA_OLD     = 0.0     # past-class regularizer (0 in the reference Bios scripts)
FSW_TAU            = 4.379673760750782     # replay gradient weight: g <- g_curr + tau * g_mem
FSW_DROP_THRESHOLD = 0.05    # samples below this weight are dropped for the epoch

SEED = 42

import os
import torch as _torch
DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"

# Spurious Shift — 6 tasks with escalating spurious correlation: 0.0→0.5→1.5→2.5→3.0→4.0
# Group 0 (80%): shortcut helps. Group 1 (20%): shortcut hurts.
# Two rotating legitimate features prevent self-correction across the task arc.
SPURIOUS_N_TASKS       = 6    # train.py task-count fallback in spurious mode
SPURIOUS_SEED          = 42   # factory.get_tasks() spurious branch default

# FaIRL
FAIRL_BETA          = 1.0
FAIRL_GAMMA         = 0.5
FAIRL_ETA           = 0.5
FAIRL_EXEMPLAR_SIZE = 800
FAIRL_LAMBDA        = 0.1661406088171958
FAIRL_EPS           = 1.989138708576197
FAIRL_ADV_DIM       = 64
FAIRL_D_STEP        = 5

# Bias-in-Bios
BIOS_DATA_PATH        = os.environ.get("BIOS_DATA_PATH", "../data/bios")
BIOS_STEP_SIZE        = 5       # professions per CL task → 5 tasks total
BIOS_BERT_BATCH       = 64
BIOS_BERT_MAXLEN      = 32
BIOS_BERT_LR          = 3.4088e-05  # fallback for non-study code paths

# Per-method BERT LRs — each method keeps its own so tune.py doesn't overwrite others
JOINT_BERT_LR         = 3.674059202635225e-05
FINETUNE_BERT_LR      = 7.160849144555761e-06
AGEM_BERT_LR          = 4.5477503819816614e-05  # tune.py --methods A-GEM overwrites this
GROUPDRO_BERT_LR      = 5.0553451864456875e-06
DERPP_BERT_LR         = 2.6975154833351135e-05
FAIRL_BERT_LR         = 1.6557979750342468e-05
FAIRGEM_BERT_LR       = 2.045610287221892e-05
FAIRDERPP_BERT_LR     = 5.081572060107125e-06
FSW_BERT_LR           = 3.7895176228853744e-05  # tune.py --methods FSW overwrites this
BIOS_BERT_TRAIN_BATCH = 256
BIOS_BERT_EPOCHS      = 2
BIOS_BERT_FINETUNE    = True   # True → fine-tune BERT end-to-end; False → use cached embeddings
