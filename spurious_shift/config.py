"""config.py — stores all hyperparams from tuning.
"""

N_CLASSES = 2

# MLP (shared by all neural methods)
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

# Episodic memory (A-GEM, FAIR-GEM)
MEMORY_SIZE_PER_TASK = 800
MEMORY_STRATIFIED    = True
AGEM_MEM_BATCH_SIZE  = 448

# FSW (Park et al. 2026) — EO variant, domain-IL adaptation (see fsw.py)
FSW_ALPHA          = 0.0010754441064902479   # gradient-adjustment scale in the linearized group loss
FSW_LAMBDA         = 0.286595105263159     # class-loss (accuracy) regularizer
FSW_TAU            = 1.4417041209859551     # replay gradient weight: g <- g_curr + tau * g_mem
FSW_DROP_THRESHOLD = 0.05    # samples below this weight are dropped for the epoch

# SGD-projected ablation arms (exact per-step guarantee; plain SGD)
FAIRGEM_SGD_LR   = 0.0030   # tune.py --methods FAIR-GEM-SGD overwrites
FAIRDERPP_SGD_LR = 0.0226   # tune.py --methods Fair-DER++-SGD overwrites

# DER++ (tuned on Spurious Shift)
DERPP_ALPHA     = 0.1724
DERPP_BETA      = 0.1732
DERPP_FAIR_LAMBDA = 1.0  # tune.py --methods DER++-Penalty overwrites
DERPP_MEM_SIZE  = 800
DERPP_MEM_BATCH = 448

# FAIRL (tuned on Spurious Shift)
FAIRL_LAMBDA    = 4.1749
FAIRL_EPS       = 1.3385
FAIRL_ADV_DIM   = 64
FAIRL_MEM_SIZE  = 800
FAIRL_MEM_BATCH = 448

# GroupDRO (tuned on Spurious Shift)
DRO_ETA = 0.0056
DRO_ADJ = 2.8521

# FAIR-GEM
# eps_t = min(alpha, beta * L_DRO) where L_DRO = L_worst - L_best.
FAIRGEM_TOP_K     = 64
FAIRGEM_ALPHA     = 0.3794
FAIRGEM_BETA      = 3.1940
FAIRGEM_MEM_SIZE  = 800
FAIRGEM_MEM_BATCH = 448

# Fair-DER++ (tuned on Spurious Shift)
FAIRDERPP_ALPHA     = 1.0460
FAIRDERPP_BETA      = 0.2601
FAIRDERPP_EPS_ALPHA = 0.3753
FAIRDERPP_EPS_BETA  = 1.1352
FAIRDERPP_MEM_SIZE  = 800
FAIRDERPP_MEM_BATCH = 448

SEED = 42

import torch as _torch
DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"

# Spurious shift dataset
SPURIOUS_N_TASKS       = 6
SPURIOUS_SEED          = 42
SPURIOUS_ANTI_STRENGTH = 0.3
