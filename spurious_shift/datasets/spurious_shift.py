"""
spurious_shift.py — Synthetic dataset designed to activate fairness gradient projection.

- Feature 0 (f0): rotating legitimate signal. active on tasks 0-2, inactive on tasks 3-5
- Feature 1 (f1): rotating legitimate signal. inactive on tasks 0-2, active on tasks 3-5
- Feature 2 (f2): spurious shortcut. positive for majority (Group 0), negative for minority
- Features 3-19: noise
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

from config import SPURIOUS_ANTI_STRENGTH


N_FEATURES = 20
MAJORITY_FRAC = 0.80        # fraction of samples that are Group 0 (majority)
N_TRAIN = 4_000
# The fairness metrics are max-min gaps over per-group accuracies, so their
# sampling error is driven by the *minority* group (20% of the split).
# N_TEST=10_000 gives ~2_000 minority samples, a 95% CI of about +/-0.01 on a
# reported gap.
#
# N_VAL must keep the minority group comfortably above the metrics'
# min_group_size threshold; otherwise accuracy_parity_gap returns its
# <2-groups sentinel and the tuning objective reduces to balanced accuracy.
# N_VAL=2_000 gives ~400 minority samples (gap CI roughly +/-0.05).
N_VAL = 2_000
N_TEST = 10_000


def _make_task(
    shortcut_strength: float,
    active_legit_idx: int,          # 0 or 1: which legit feature is "on"
    legit_strength: float,          # positive signal strength for the active feature
    anti_strength: float = SPURIOUS_ANTI_STRENGTH,   # negative signal on the inactive feature
    noise_std: float = 1.0,
    seed: int = 0,
) -> dict:
    """
    Generate a single task.
    """
    # One independent stream per field. With a single stream, changing an
    # evaluation size would shift the stream position for every later field
    # and so change the training rows. Spawning per field makes the first
    # N_TRAIN rows of each field invariant to N_VAL and N_TEST.
    root = np.random.default_rng(seed)
    r_s, r_y, r_act, r_inact, r_f2, r_noise, r_split = root.spawn(7)

    n = N_TRAIN + N_VAL + N_TEST
    s = r_s.choice([0, 1], size=n, p=[MAJORITY_FRAC, 1 - MAJORITY_FRAC])
    y = r_y.integers(0, 2, size=n)

    label_signal = (2 * y - 1).astype(np.float64)  # +1 for y=1, -1 for y=0

    f_active   =  legit_strength * label_signal + r_act.normal(0, noise_std, n)
    f_inactive = -anti_strength  * label_signal + r_inact.normal(0, noise_std, n)

    if active_legit_idx == 0:
        f0, f1 = f_active, f_inactive
    else:
        f0, f1 = f_inactive, f_active

    sign = np.where(s == 0, +1, -1)
    f2 = shortcut_strength * sign * label_signal + r_f2.normal(0, noise_std, n)

    noise = r_noise.normal(0, noise_std, (n, N_FEATURES - 3))

    X = np.column_stack([f0, f1, f2, noise]).astype(np.float32)

    # Split into train, val, test.
    #
    # The training rows are taken from the head of the array, not from a
    # permutation of all n, so the training set does not depend on the
    # evaluation sizes. Samples are i.i.d. by construction, so a contiguous
    # split is as valid as a shuffled one.
    train_idx = np.arange(N_TRAIN)
    eval_idx  = N_TRAIN + r_split.permutation(N_VAL + N_TEST)
    # Test rows are drawn first so the test set does not move when N_VAL alone
    # is adjusted (the permutation length still ties both together).
    test_idx  = eval_idx[:N_TEST]
    val_idx   = eval_idx[N_TEST:]

    return {
        "X": X,
        "y": y.astype(np.int64),
        "s": s.astype(np.int64),
        "train_idx": train_idx,
        "val_idx":   val_idx,
        "test_idx":  test_idx,
    }


def get_spurious_tasks(seed: int = 42) -> tuple:
    """
    Returns (tasks, scaler)
    """
    shortcut_schedule    = [0.0, 0.5, 1.5, 2.5, 3.0, 4.0]
    legit_schedule       = [2.0, 1.8, 1.5, 1.2, 1.0, 0.8]
    active_legit_schedule = [0, 0, 0, 1, 1, 1]

    raw_tasks = [
        _make_task(
            shortcut_strength=shortcut_schedule[i],
            active_legit_idx=active_legit_schedule[i],
            legit_strength=legit_schedule[i],
            # Disjoint per-task roots: seed+i would make run N and run N+1
            # share 5 of 6 task RNG streams, correlating the "independent"
            # seeds and understating cross-seed variance.
            seed=seed * 1000 + i,
        )
        for i in range(len(shortcut_schedule))
    ]

    # Fit scaler on task 0 training data only (feat_0 active at 2.0, shortcut=0 — cleanest task)
    scaler = StandardScaler().fit(raw_tasks[0]["X"][raw_tasks[0]["train_idx"]])

    tasks = []
    for i, (raw, sc, ls, al) in enumerate(
        zip(raw_tasks, shortcut_schedule, legit_schedule, active_legit_schedule)
    ):
        X = scaler.transform(raw["X"])
        ti, vi, ei = raw["train_idx"], raw["val_idx"], raw["test_idx"]
        active_name = f"f{al}"
        tasks.append({
            "state":   f"task_{i}__active={active_name}__legit={ls}__shortcut={sc}",
            "X_train": X[ti], "y_train": raw["y"][ti], "s_train": raw["s"][ti],
            "X_val":   X[vi], "y_val":   raw["y"][vi], "s_val":   raw["s"][vi],
            "X_test":  X[ei], "y_test":  raw["y"][ei], "s_test":  raw["s"][ei],
        })

    return tasks, scaler
