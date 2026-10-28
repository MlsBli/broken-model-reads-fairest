"""
metrics.py — Evaluation metrics for accuracy and fairness.

Fairness metrics (lower is fairer; 0 = perfect parity):
  tpr_gap_rms            — per-profession TPR gap, RMS-aggregated (PRIMARY; De-Arteaga 2019)
  equalized_odds_gap     — max(RMS TPR gap, RMS FPR gap), one-vs-rest per class
  demographic_parity_gap — sum_p |P(ŷ=p|g=0) − P(ŷ=p|g=1)| (FaIRL formulation)
  accuracy_parity_gap    — max − min per-group macro-recall
  equal_opportunity_gap  — binary TPR parity only (reference)

Performance-equity (higher is better):
  worst_group_balanced_accuracy — min over groups of per-group balanced accuracy

CL metrics:
  backward_transfer (BWT), fairness_backward_transfer (F-BWT), forward_transfer (FWT)

Groups with fewer than MIN_GROUP_SIZE samples are excluded from all fairness metrics.
"""
# NOTE: these are multi-class metrics — EO/TPR gaps are RMS one-vs-rest with
# MIN_GROUP_SIZE=20. The Spurious Shift suite uses binary max-gaps with
# min_group_size=100 under the same names, so raw values are not comparable
# across benchmarks.

import numpy as np
from sklearn.metrics import recall_score

MIN_GROUP_SIZE: int = 20


def task_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_pred == y_true).mean())



def accuracy_parity_gap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    min_group_size: int = MIN_GROUP_SIZE,
) -> float:
    """Max − min per-group macro-recall across sensitive groups.

    Restricts to classes present in y_true to avoid phantom zeros from
    classes absent in the current task. Lower is fairer.
    """
    labels = np.unique(y_true).tolist()
    groups = np.unique(sensitive)
    accs = []
    for g in groups:
        mask = sensitive == g
        if mask.sum() >= min_group_size:
            accs.append(recall_score(
                y_true[mask], y_pred[mask],
                labels=labels, average="macro", zero_division=0,
            ))
    # <2 measurable groups: the gap is UNDEFINED, not 0.0 — zero is the best
    # possible value and would book missing data as perfect fairness.
    return float(max(accs) - min(accs)) if len(accs) >= 2 else float("nan")


def equalized_odds_gap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    min_group_size: int = MIN_GROUP_SIZE,
) -> float:
    """Equalized odds gap (Hardt et al., 2016), generalised to multi-class via
    one-vs-rest. Returns max(RMS TPR gap, RMS FPR gap) across all classes.
    Lower is fairer; by construction EO >= TPR gap.
    """
    groups = np.unique(sensitive)
    if len(groups) < 2:
        return float("nan")
    g0, g1 = groups[0], groups[1]

    classes = np.unique(y_true)
    tpr_gaps, fpr_gaps = [], []
    for p in classes:
        pos = (y_true == p)
        neg = ~pos
        t0, t1 = pos & (sensitive == g0), pos & (sensitive == g1)
        if t0.sum() >= min_group_size and t1.sum() >= min_group_size:
            tpr_gaps.append(float((y_pred[t0] == p).mean() - (y_pred[t1] == p).mean()))
        f0, f1 = neg & (sensitive == g0), neg & (sensitive == g1)
        if f0.sum() >= min_group_size and f1.sum() >= min_group_size:
            fpr_gaps.append(float((y_pred[f0] == p).mean() - (y_pred[f1] == p).mean()))

    if not tpr_gaps and not fpr_gaps:
        return float("nan")      # no class had both groups measurable
    tpr_rms = float(np.sqrt(np.mean(np.square(tpr_gaps)))) if tpr_gaps else 0.0
    fpr_rms = float(np.sqrt(np.mean(np.square(fpr_gaps)))) if fpr_gaps else 0.0
    return max(tpr_rms, fpr_rms)



def per_group_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    min_group_size: int = MIN_GROUP_SIZE,
) -> dict:
    """Balanced accuracy per group.

    Uses recall_score(labels=present_classes) so absent classes don't inflate
    the gap with phantom zero-recall scores.
    """
    groups = np.unique(sensitive)
    result = {}
    for g in groups:
        mask = sensitive == g
        if mask.sum() < min_group_size:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        present = np.unique(yt)
        result[int(g)] = float(
            recall_score(yt, yp, labels=present, average="macro", zero_division=0)
        )
    return result


def worst_group_balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    min_group_size: int = MIN_GROUP_SIZE,
) -> float:
    """Min over sensitive groups of per-group balanced accuracy.

    The Rawlsian / DRO min-max objective that FAIR-GEM and Fair-DER++ target.
    Higher is better; returns NaN if no group clears min_group_size.
    """
    pg = per_group_accuracy(y_true, y_pred, sensitive, min_group_size)
    if not pg:
        return float("nan")
    return float(min(pg.values()))


def demographic_parity_gap(
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    min_group_size: int = MIN_GROUP_SIZE,
) -> float:
    """Demographic parity gap: sum_p |P(ŷ=p | g=0) − P(ŷ=p | g=1)|.

    FaIRL (Basu Roy Chowdhury & Chaturvedi, AAAI 2023) formulation for multi-class prediction.
    Lower is fairer.
    """
    groups = np.unique(sensitive)
    eligible = [g for g in groups if (sensitive == g).sum() >= min_group_size]
    if len(eligible) < 2:
        return float("nan")      # undefined, not "perfectly fair"

    all_classes = np.unique(y_pred)
    dp = 0.0
    for p in all_classes:
        rates = []
        for g in eligible:
            mask = sensitive == g
            rates.append(float((y_pred[mask] == p).mean()))
        dp += max(rates) - min(rates)
    return float(dp)


def tpr_gap_rms(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    min_group_size: int = MIN_GROUP_SIZE,
) -> float:
    """Per-profession TPR gap, RMS-aggregated (De-Arteaga et al. 2019; FaIRL AAAI 2023).

        TPR_{p,g} = P(ŷ=p | y=p, group=g)
        GAP_RMS   = sqrt(mean_p (TPR_{p,g0} − TPR_{p,g1})^2)

    A profession p contributes only if both groups have >= min_group_size true samples.
    Lower is fairer.
    """
    groups = np.unique(sensitive)
    if len(groups) < 2:
        return float("nan")
    g0, g1 = groups[0], groups[1]

    professions = np.unique(y_true)
    gaps = []
    for p in professions:
        true_p = (y_true == p)
        m0 = true_p & (sensitive == g0)
        m1 = true_p & (sensitive == g1)
        if m0.sum() < min_group_size or m1.sum() < min_group_size:
            continue
        tpr0 = float((y_pred[m0] == p).mean())
        tpr1 = float((y_pred[m1] == p).mean())
        gaps.append(tpr0 - tpr1)

    if not gaps:
        return float("nan")      # no profession had both groups measurable
    return float(np.sqrt(np.mean(np.square(gaps))))


def fairness_backward_transfer(fair_matrix: np.ndarray) -> float:
    """F-BWT = mean(Fair_gap(i,i) - Fair_gap(T-1,i)) for i < T-1.

    Negative = fairness was forgotten (gap grew on old tasks).
    Positive = fairness improved on old tasks after further training.
    """
    T = fair_matrix.shape[0]
    if T < 2:
        return float("nan")
    fbwt = 0.0
    count = 0
    for i in range(T - 1):
        if not np.isnan(fair_matrix[T - 1, i]) and not np.isnan(fair_matrix[i, i]):
            fbwt += fair_matrix[i, i] - fair_matrix[T - 1, i]
            count += 1
    return float(fbwt / count) if count > 0 else float("nan")


def backward_transfer(acc_matrix: np.ndarray) -> float:
    """BWT = mean(R[T-1, i] - R[i, i]) for i < T-1.

    R[t, i] = accuracy on task i after training on task t.
    Negative = forgetting; positive = backward plasticity.
    """
    T = acc_matrix.shape[0]
    if T < 2:
        return float("nan")
    bwt = 0.0
    count = 0
    for i in range(T - 1):
        if not np.isnan(acc_matrix[T - 1, i]) and not np.isnan(acc_matrix[i, i]):
            bwt += acc_matrix[T - 1, i] - acc_matrix[i, i]
            count += 1
    return float(bwt / count) if count > 0 else float("nan")



SUMMARY_HEADER = (f"  {'Method':30s}  {'BalAcc':>7}  {'WrstGrp':>7}  {'TPR':>6}  "
                  f"{'EO':>6}  {'DP':>6}  {'AccPar':>6}  {'BWT':>7}  {'F-BWT':>7}")


def summarize(acc_matrix: np.ndarray, bal_acc_matrix: np.ndarray,
              worst_group_matrix: np.ndarray, tpr_matrix: np.ndarray,
              eo_matrix: np.ndarray, dp_matrix: np.ndarray,
              acc_parity_matrix: np.ndarray, method_name: str):
    """Print a one-line summary for a method after all tasks are complete."""
    T = acc_matrix.shape[0]
    final_bal_acc = np.nanmean(bal_acc_matrix[T - 1, :T])
    final_worst   = np.nanmean(worst_group_matrix[T - 1, :T])
    final_tpr     = np.nanmean(tpr_matrix[T - 1, :T])
    final_eo      = np.nanmean(eo_matrix[T - 1, :T])
    final_dp      = np.nanmean(dp_matrix[T - 1, :T])
    final_accpar  = np.nanmean(acc_parity_matrix[T - 1, :T])
    bwt           = backward_transfer(bal_acc_matrix)
    # Same matrix as aggregate.py (eo), so stdout and the summary CSV agree.
    f_bwt         = fairness_backward_transfer(eo_matrix)
    print(f"  {method_name:30s}  {final_bal_acc:7.3f}  {final_worst:7.3f}  "
          f"{final_tpr:6.3f}  {final_eo:6.3f}  {final_dp:6.3f}  "
          f"{final_accpar:6.3f}  {bwt:+7.3f}  {f_bwt:+7.3f}")


def group_loss_disparity(losses, sensitive, min_group_size: int = MIN_GROUP_SIZE) -> float:
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
                                     min_group_size: int = MIN_GROUP_SIZE) -> float:
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
