"""
metrics.py — Evaluation metrics for the paper.

balanced_accuracy (accounts for class imbalance)
accuracy_parity_gap (max - min per-group balanced accuracy)
backward_transfer, fairness_backward_transfer
top_class_rate_gap (demographic parity proxy)

Groups with < min_group_size samples (default 100) are excluded from fairness metrics.
"""
# Differs from bias_in_bios/metrics.py: this suite is binary classification
# with min_group_size=100 (10k test split); Bias-in-Bios is 25-class with
# MIN_GROUP_SIZE=20 and RMS-aggregated one-vs-rest gaps. The metric names
# overlap but the formulas differ, so raw values are not comparable across
# benchmarks.

import numpy as np
from sklearn.metrics import balanced_accuracy_score, recall_score
from config import N_CLASSES


def task_accuracy(y_true, y_pred):
    return float((y_pred == y_true).mean())



def accuracy_parity_gap(y_true, y_pred, sensitive, min_group_size=100):
    """
    Max - min per-group balanced accuracy across demographic groups.
    """
    groups = np.unique(sensitive)
    accs = [balanced_accuracy_score(y_true[sensitive == g], y_pred[sensitive == g])
              for g in groups if (sensitive == g).sum() >= min_group_size]
    # Fewer than 2 measurable groups means the gap is undefined, not zero.
    return float(max(accs) - min(accs)) if len(accs) >= 2 else float("nan")


def equalized_odds_gap(y_true, y_pred, sensitive, min_group_size=100):
    """
    Worst-case equalized odds gap.
    """
    groups = np.unique(sensitive)
    tprs, fprs = [], []
    for g in groups:
        pos_mask = (sensitive == g) & (y_true == 1)
        neg_mask = (sensitive == g) & (y_true == 0)
        if pos_mask.sum() >= min_group_size:
            tprs.append((y_pred[pos_mask] == 1).mean())
        if neg_mask.sum() >= min_group_size:
            fprs.append((y_pred[neg_mask] == 1).mean())
    if len(tprs) < 2 or len(fprs) < 2:
        return float("nan")     # undefined, not "perfectly fair"
    return max(float(max(tprs) - min(tprs)), float(max(fprs) - min(fprs)))



def per_group_accuracy(y_true, y_pred, sensitive, min_group_size=100):
    """
    Balanced accuracy per group.
    """
    groups = np.unique(sensitive)
    result = {}
    for g in groups:
        mask = sensitive == g
        if mask.sum() < min_group_size:
            continue
        yt, yp  = y_true[mask], y_pred[mask]
        present = np.unique(yt)
        result[int(g)] = float(recall_score(yt, yp, labels=present, average="macro", zero_division=0))
    return result


def top_class_rate_gap(y_pred, sensitive, min_group_size=100):
    """
    Max - min fraction predicted as the top class across groups (demographic parity proxy).
    """
    top_class = N_CLASSES - 1
    groups = np.unique(sensitive)
    rates = [(y_pred[sensitive == g] == top_class).mean()
                 for g in groups if (sensitive == g).sum() >= min_group_size]
    return float(max(rates) - min(rates)) if len(rates) >= 2 else float("nan")


# Backward transfer compares each task's accuracy at the end of training against
# its accuracy when it was first learned, so it needs both the final row and the
# diagonal of the lower-triangular matrix. When the diagonal is absent (as in
# results_test.npz, where only the final row is populated) the quantity is
# undefined and NaN is returned.
def backward_transfer(acc_matrix):
    """
    Backward transfer: negative -> forgetting, positive -> not forgetting.

    NaN when the matrix carries no diagonal to compare the final row against.
    """
    T = acc_matrix.shape[0]
    if T < 2:
        return float("nan")
    vals = [acc_matrix[T-1, i] - acc_matrix[i, i]
            for i in range(T-1)
            if not np.isnan(acc_matrix[T-1, i]) and not np.isnan(acc_matrix[i, i])]
    return float(np.mean(vals)) if vals else float("nan")


def fairness_backward_transfer(fair_matrix):
    """
    Equivalent to BWT but for fairness. NaN when the diagonal is absent.
    """
    T = fair_matrix.shape[0]
    if T < 2:
        return float("nan")
    vals = [fair_matrix[i, i] - fair_matrix[T-1, i]
            for i in range(T-1)
            if not np.isnan(fair_matrix[T-1, i]) and not np.isnan(fair_matrix[i, i])]
    return float(np.mean(vals)) if vals else float("nan")


def summarize(acc_matrix, bal_acc_matrix, fair_matrix, dp_matrix, method_name,
              eo_true_matrix=None, worst_group_matrix=None):
    """
    One summary line per method.

    `fair_matrix` is the accuracy-parity gap — the quantity the paper reports as
    its fairness metric. When supplied, `eo_true_matrix` (textbook equalized-odds
    gap) and `worst_group_matrix` (worst per-group balanced accuracy) are printed
    beside it, so a small disparity reached by collapsing toward chance can be
    told apart from one reached by closing the gap.
    """
    T = acc_matrix.shape[0]
    final_bal_acc = np.nanmean(bal_acc_matrix[T-1, :T])
    final_fair = np.nanmean(fair_matrix[T-1, :T])
    final_dp = np.nanmean(dp_matrix[T-1, :T])
    # "n/a" rather than "+nan": the metric is undefined for this matrix (no
    # diagonal), which is a different statement from a value that came out NaN.
    def _signed(x):
        return "    n/a" if np.isnan(x) else f"{x:+.3f}"

    bwt = backward_transfer(bal_acc_matrix)
    f_bwt = fairness_backward_transfer(fair_matrix)
    line = (f"  {method_name:30s}  bal={final_bal_acc:.3f}  AccParity={final_fair:.3f}"
            f"  TopRate={final_dp:.3f}  BWT={_signed(bwt)}  F-BWT={_signed(f_bwt)}")
    if eo_true_matrix is not None:
        line += f"  EO={np.nanmean(eo_true_matrix[T-1, :T]):.3f}"
    if worst_group_matrix is not None:
        line += f"  worst={np.nanmean(worst_group_matrix[T-1, :T]):.3f}"
    print(line)

def group_loss_disparity(losses, sensitive, min_group_size: int = 100) -> float:
    """CE-surrogate gap: max_g mean loss - min_g mean loss (unconditional).

    The differentiable group-loss disparity that the projection methods (and
    FSW's LP) actually steer on, measured at eval time so its alignment with
    the prediction-space fairness metrics can be validated empirically.
    """
    losses = np.asarray(losses); sensitive = np.asarray(sensitive)
    means = [float(losses[sensitive == g].mean()) for g in np.unique(sensitive)
             if (sensitive == g).sum() >= min_group_size]
    if len(means) < 2:
        return float("nan")
    return max(means) - min(means)


def conditional_group_loss_disparity(losses, y_true, sensitive,
                                     min_group_size: int = 100) -> float:
    """Label-conditioned surrogate: mean over classes y of (max_g - min_g) of
    mean loss in cell (y, g). Cells under min_group_size are skipped; NaN if no
    class has two eligible cells. More naturally aligned with EO-style criteria.
    """
    losses = np.asarray(losses); y_true = np.asarray(y_true); sensitive = np.asarray(sensitive)
    gaps = []
    for c in np.unique(y_true):
        m = y_true == c
        means = [float(losses[m & (sensitive == g)].mean()) for g in np.unique(sensitive)
                 if int((m & (sensitive == g)).sum()) >= min_group_size]
        if len(means) >= 2:
            gaps.append(max(means) - min(means))
    return float(np.mean(gaps)) if gaps else float("nan")
