"""
fairderpp.py — Fair-DER++: DER++ with a disparity-safety gradient projection.

Combines DER++ logit-distillation replay with a single fairness projection:

    L = CE(f(x_t), y_t)
      + alpha * MSE(f(x_mem), z_mem)     [distill past logits]
      + beta  * CE(f(x_mem), y_mem)      [reinforce past labels]

After back-prop, project to satisfy:
    dot(g, g_safety_n) >= eps_t
    eps_t = min(eps_alpha, eps_beta * L_DRO)   [scales with current disparity]

DER++ handles memory retention through the loss terms, so only one projection
constraint is needed (unlike FAIR-GEM's cyclic dual projection).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from memory import EpisodicMemory
from model import (MLP, make_model, make_optimizer, make_scheduler, has_converged,
                   predict, mask_logits, _x_tensor, _train_batch_size,
                   batchnorm_eval)
from baselines_neural import (_clip_flat_grad, _apply_step_with_diag,
                              _diag_sink_or_none, _realized_stats)
from config import (
    MLP_EPOCHS, MLP_BATCH_SIZE, MLP_GRAD_CLIP, DEVICE,
    AGEM_MEM_BATCH_SIZE,
    FAIRDERPP_ALPHA, FAIRDERPP_BETA,
    FAIRDERPP_EPS_ALPHA, FAIRDERPP_EPS_BETA, SEED,
)
import runctx
import config
from safety import conditional_disparity, validate_disparity_mode


def _make_loader(dataset, batch_size, shuffle=True):
    g = torch.Generator(); g.manual_seed(runctx.effective_seed())  # --seed varies batch order
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g)


def _mode_or_nan(values):
    if not values:
        return float("nan")
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[np.argmax(counts)])


def _switch_rate(values):
    if len(values) < 2:
        return 0.0
    return float(np.mean(np.asarray(values[1:]) != np.asarray(values[:-1])))


def _diag_stats(loss_gaps, worst_groups, best_groups):
    q = np.percentile(loss_gaps, [10, 50, 90]) if loss_gaps else [0.0, 0.0, 0.0]
    return {
        "loss_gap_mean": float(np.mean(loss_gaps)) if loss_gaps else 0.0,
        "loss_gap_std": float(np.std(loss_gaps)) if loss_gaps else 0.0,
        "loss_gap_p10": float(q[0]),
        "loss_gap_p50": float(q[1]),
        "loss_gap_p90": float(q[2]),
        "worst_group_mode": _mode_or_nan(worst_groups),
        "best_group_mode": _mode_or_nan(best_groups),
        "worst_group_switch_rate": _switch_rate(worst_groups),
        "best_group_switch_rate": _switch_rate(best_groups),
    }


class FairDERpp:
    """Fair-DER++: DER++ replay + disparity-safety gradient projection.

    Memory stores (X, y, s, logits). The DER++ loss handles continual learning;
    a fairness projection on top keeps the update from increasing the
    worst-group−best-group loss disparity.
    """

    def __init__(
        self,
        input_dim: int,
        device: str = DEVICE,
        alpha: float = FAIRDERPP_ALPHA,
        beta: float = FAIRDERPP_BETA,
        eps_alpha: float = FAIRDERPP_EPS_ALPHA,
        eps_beta: float = FAIRDERPP_EPS_BETA,
        bert_mode: bool = False,
        bert_lr: float = None,
        disparity_mode: str = "unconditional",
    ):
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.eps_alpha = eps_alpha
        self.eps_beta = eps_beta
        self.current_eps = eps_alpha
        self.bert_mode = bert_mode
        self.bert_lr = bert_lr
        self.disparity_mode = validate_disparity_mode(disparity_mode)

        self.model = make_model(input_dim, bert_mode).to(device)
        self.optimizer = make_optimizer(self.model, bert_lr=bert_lr)
        self.memory = EpisodicMemory(device=device)


    def _safety_gradient_conditional(self, xb, yb, sb):
        """Gradient of the mean estimable within-label group-loss gap."""
        self._last_safety_diag = None
        self.optimizer.zero_grad()
        with batchnorm_eval(self.model):
            logits = mask_logits(self.model(xb), self._seen_classes)
            per_sample = nn.CrossEntropyLoss(reduction="none")(logits, yb)
            disparity, n_labels = conditional_disparity(per_sample, yb, sb)
        if disparity is None:
            return None, 0.0
        val = float(disparity.detach().item())
        self._last_safety_diag = {
            "loss_gap": val,
            "conditioned_labels": n_labels,
        }
        disparity.backward()
        return self.model.get_flat_grad().clone(), val

    def _safety_gradient_disparity(self, xb: torch.Tensor, yb: torch.Tensor,
                                    sb: torch.Tensor,
                                    mem_batch: tuple | None = None) -> tuple[torch.Tensor | None, float]:
        """Disparity safety gradient: g_worst_group − g_best_group.

        Single backward pass via loss = L_worst − L_best. Cancels legitimate-
        feature agreement between groups, isolating the spurious-feature conflict.
        """
        self._last_safety_diag = None
        if mem_batch is not None:
            mx, my, ms = mem_batch
            xb = torch.cat([xb, mx])
            yb = torch.cat([yb, my])
            sb = torch.cat([sb, ms])

        groups = sb.unique()
        self.optimizer.zero_grad()
        with batchnorm_eval(self.model):
            logits = mask_logits(self.model(xb), self._seen_classes)
        per_sample = nn.CrossEntropyLoss(reduction="none")(logits, yb)

        group_losses: dict[int, torch.Tensor] = {}
        for g in groups:
            gmask = sb == g
            if gmask.sum() < 2:
                continue
            group_losses[g.item()] = per_sample[gmask].mean()

        if len(group_losses) < 2:
            # One group in the batch: disparity is undefined; skip the safety
            # constraint rather than descend the lone group's loss at a
            # raw-loss-sized eps.
            return None, 0.0

        # --- Optional EMA smoothing of the disparity direction ---
        # On BIOS the per-batch worst-group identity flips on ~45% of steps:
        # the batch-level gender loss gap is smaller than its sampling noise,
        # so successive safety projections can point in near-opposite
        # directions. Ranking groups by an EMA of their losses
        # (config.SAFETY_EMA, 0 disables) stabilises the direction; the
        # gradient is still taken on current-batch losses, and the eps
        # schedule sees the smoothed gap.
        ema_m = float(getattr(config, "SAFETY_EMA", 0.0))
        rank_vals = {g: l.item() for g, l in group_losses.items()}
        if ema_m > 0.0:
            if not hasattr(self, "_ema_group_loss"):
                self._ema_group_loss = {}
            for g, v in rank_vals.items():
                prev = self._ema_group_loss.get(g, v)
                self._ema_group_loss[g] = ema_m * prev + (1.0 - ema_m) * v
            rank_vals = {g: self._ema_group_loss[g] for g in rank_vals}

        sorted_gs = sorted(rank_vals, key=rank_vals.get)
        best_g, worst_g = sorted_gs[0], sorted_gs[-1]
        sched_gap = rank_vals[worst_g] - rank_vals[best_g]

        disparity_loss = group_losses[worst_g] - group_losses[best_g]
        self._last_safety_diag = {
            "best_group": int(best_g),
            "worst_group": int(worst_g),
            "loss_gap": float(disparity_loss.detach().item()),
        }
        disparity_loss.backward()
        return self.model.get_flat_grad().clone(), float(sched_gap)

    def _project_fairness(self, g: torch.Tensor, g_safety: torch.Tensor | None,
                           eps: float) -> torch.Tensor:
        """Project g onto the half-space dot(g, g_safety_n) >= eps.

        eps > 0 → update must actively reduce disparity each step.
        eps = 0 → update merely does not increase disparity.
        """
        if g_safety is None:
            return g
        g_safety_n = g_safety / (g_safety.norm() + 1e-8)
        dot = (g * g_safety_n).sum()
        if dot < eps:
            g = g - (dot - eps) * g_safety_n
        return g

    def train_task(self, X_train: np.ndarray, y_train: np.ndarray,
                   s_train: np.ndarray, epochs: int = MLP_EPOCHS,
                   seen_classes: list[int] | None = None):
        self._seen_classes = seen_classes
        X_t = _x_tensor(X_train)
        y_t = torch.tensor(y_train, dtype=torch.long)
        s_t = torch.tensor(s_train, dtype=torch.long)
        loader = _make_loader(
            TensorDataset(X_t, y_t, s_t),
            batch_size=_train_batch_size(self.bert_mode),
        )

        scheduler = make_scheduler(self.optimizer, epochs, steps_per_epoch=len(loader))
        self.model.train()
        epoch_losses = []

        _n_steps       = 0
        _n_safety_proj = 0
        _dro_losses:   list[float] = []
        _eps_vals:     list[float] = []
        _dot_safety:   list[float] = []
        _loss_gaps:    list[float] = []
        _worst_groups: list[int] = []
        _best_groups:  list[int] = []
        _diag_sink = _diag_sink_or_none()

        for _ in range(epochs):
            batch_loss = 0.0
            for xb, yb, sb in loader:
                xb, yb, sb = xb.to(self.device), yb.to(self.device), sb.to(self.device)

                self.optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(mask_logits(self.model(xb), seen_classes), yb)

                # Two independent memory batches, matching Buzzega et al. and the
                # DER++ baseline.
                mem_distill = self.memory.get_random_batch_with_logits(AGEM_MEM_BATCH_SIZE)
                mem_replay = self.memory.get_random_batch_with_logits(AGEM_MEM_BATCH_SIZE)
                if mem_distill is not None:
                    xd, _yd, _sd, zd = mem_distill
                    xr, yr, _sr, _zr = mem_replay
                    with batchnorm_eval(self.model):
                        logits_d = self.model(xd)
                        logits_r = self.model(xr)
                    if seen_classes is not None:
                        mse = F.mse_loss(logits_d[:, seen_classes], zd[:, seen_classes])
                    else:
                        mse = F.mse_loss(logits_d, zd)
                    loss = (loss
                            + self.alpha * mse
                            + self.beta  * nn.CrossEntropyLoss()(mask_logits(logits_r, seen_classes), yr))

                loss.backward()
                g_der = self.model.get_flat_grad().clone()
                batch_loss += loss.item()

                if self.disparity_mode == "conditional":
                    g_safety, dro_loss = self._safety_gradient_conditional(xb, yb, sb)
                else:
                    g_safety, dro_loss = self._safety_gradient_disparity(xb, yb, sb)
                diag = getattr(self, "_last_safety_diag", None)
                if diag is not None:
                    _loss_gaps.append(diag["loss_gap"])
                    if "worst_group" in diag:
                        _worst_groups.append(diag["worst_group"])
                        _best_groups.append(diag["best_group"])
                self.current_eps = min(self.eps_alpha, self.eps_beta * dro_loss)

                _n_steps += 1
                _dro_losses.append(dro_loss)
                _eps_vals.append(self.current_eps)
                if g_safety is not None:
                    g_sn = g_safety / (g_safety.norm() + 1e-8)
                    dot_s = float((g_der * g_sn).sum().item())
                    _dot_safety.append(dot_s)
                    if dot_s < self.current_eps:
                        _n_safety_proj += 1

                g_proj = self._project_fairness(g_der, g_safety, self.current_eps)
                g_clip = _clip_flat_grad(g_proj, MLP_GRAD_CLIP)
                self.model.set_flat_grad(g_clip)
                if _diag_sink is not None and _n_steps % 4 == 1:
                    _apply_step_with_diag(self.model, self.optimizer, g_clip,
                                          g_safety, None, self.current_eps, _diag_sink)
                else:
                    self.optimizer.step()
                scheduler.step()
            epoch_losses.append(batch_loss / len(loader))
            if has_converged(epoch_losses):
                break

        self.proj_stats: dict = {
            "safety_proj_rate": _n_safety_proj / _n_steps if _n_steps else 0.0,
            "mean_dro_loss":    float(np.mean(_dro_losses)) if _dro_losses else 0.0,
            "mean_eps_t":       float(np.mean(_eps_vals))   if _eps_vals   else 0.0,
            "mean_dot_safety":  float(np.mean(_dot_safety)) if _dot_safety else 0.0,
            "n_steps":          _n_steps,
        }
        self.proj_stats.update(_diag_stats(_loss_gaps, _worst_groups, _best_groups))
        self.proj_stats.update(_realized_stats(_diag_sink))

    def remember(self, X: np.ndarray, y: np.ndarray, s: np.ndarray):
        """Compute logits on X and add (X, y, s, logits) to memory."""
        self.model.eval()
        X_t = _x_tensor(X)
        parts = []
        with torch.no_grad():
            for i in range(0, len(X_t), 256):
                parts.append(self.model(X_t[i:i+256].to(self.device)).cpu())
        logits = torch.cat(parts)
        self.model.train()
        self.memory.add_task_with_logits(X, y, s, logits)

    def predict(self, X: np.ndarray, seen_classes: list[int] | None = None) -> np.ndarray:
        return predict(self.model, X, self.device, seen_classes=seen_classes)
