"""
fairgem.py — FAIR-GEM: fairness-aware continual learning via dual gradient projection.

Extends A-GEM with:
  1. Safety gradient g_safety = g_worst_group − g_best_group (disparity direction)
  2. Dynamic safety margin: eps_t = min(alpha, beta * L_DRO)
  3. Cyclic projection alternating between the memory and safety half-spaces

Ablation modes:
  use_topk_dro=True          → top-k DRO on worst-group samples (noisier)
  use_disparity=False        → average-group gradient (degenerates to A-GEM)
  dynamic_eps=False          → static margin = alpha
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from memory import EpisodicMemory
from model import make_scheduler, mask_logits, _x_tensor, _train_batch_size
from model import has_converged, batchnorm_eval
from baselines_neural import _apply_step_with_diag, _diag_sink_or_none, _realized_stats, AGEM, _clip_flat_grad
from config import (
    MLP_EPOCHS, MLP_BATCH_SIZE, DEVICE,
    FAIRGEM_TOP_K, FAIRGEM_ALPHA, FAIRGEM_BETA,
    MLP_GRAD_CLIP, AGEM_MEM_BATCH_SIZE, SEED,
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


class FAIRGEM(AGEM):
    """FAIR-GEM: dual gradient projection for fairness-aware continual learning.

    Targets feasibility via 15 cyclic half-space projections:
        dot(g_mem,    delta_theta) >= 0
        dot(g_safety, delta_theta) >= eps_t

    The loop has no Dykstra correction variables and does not claim the
    Euclidean-nearest point in the intersection.

    Default mode (use_topk_dro=False, use_disparity=True): disparity safety gradient
    g_worst − g_best, computed on the current batch. Scales with actual disparity so
    the projection is quiet when groups are similar and fires aggressively when they diverge.
    """
    def __init__(
        self,
        input_dim: int,
        device: str = DEVICE,
        top_k: int = FAIRGEM_TOP_K,
        alpha: float = FAIRGEM_ALPHA,
        beta: float = FAIRGEM_BETA,
        use_topk_dro: bool = False,
        use_disparity: bool = True,
        dynamic_eps: bool = True,
        proj_iters: int = 15,
        bert_mode: bool = False,
        bert_lr: float = None,
        disparity_mode: str = "unconditional",
    ):
        super().__init__(input_dim, device, bert_mode=bert_mode, bert_lr=bert_lr)
        self.top_k = top_k
        self.alpha = alpha
        self.beta = beta
        self.use_topk_dro = use_topk_dro
        self.use_disparity = use_disparity
        self.dynamic_eps = dynamic_eps
        self.proj_iters = int(proj_iters)
        if self.proj_iters < 1:
            raise ValueError("proj_iters must be at least 1")
        self.disparity_mode = validate_disparity_mode(disparity_mode)
        self.current_eps = alpha

    def _safety_gradient_topk(self, xb, yb, sb):
        sc = self._seen_classes
        criterion = nn.CrossEntropyLoss(reduction="none")
        self._last_safety_diag = None
        groups = sb.unique()
        group_losses = {}
        worst_loss, worst_group = -1.0, None

        with torch.no_grad():
            for g in groups:
                gmask = sb == g
                if gmask.sum() < 2:
                    continue
                loss = criterion(mask_logits(self.model(xb[gmask]), sc), yb[gmask]).mean().item()
                group_losses[int(g.item())] = loss
                if loss > worst_loss:
                    worst_loss, worst_group = loss, g

        if worst_group is None:
            return None, 0.0
        sorted_gs = sorted(group_losses, key=group_losses.get)
        best_g, worst_g = sorted_gs[0], sorted_gs[-1]
        self._last_safety_diag = {
            "best_group": int(best_g),
            "worst_group": int(worst_g),
            "loss_gap": float(group_losses[worst_g] - group_losses[best_g]),
        }

        gmask = sb == worst_group
        xg, yg = xb[gmask], yb[gmask]
        with torch.no_grad():
            per_sample = criterion(mask_logits(self.model(xg), sc), yg)
        k = min(self.top_k, len(per_sample))
        top_idx = per_sample.topk(k).indices

        self.optimizer.zero_grad()
        dro_loss = nn.CrossEntropyLoss()(mask_logits(self.model(xg[top_idx]), sc), yg[top_idx])
        dro_loss.backward()
        return self.model.get_flat_grad().clone(), dro_loss.item()

    def _safety_gradient_average(self, xb, yb, sb, mem_batch=None):
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
            total_loss = torch.tensor(0.0, device=self.device)
            n = 0
            for g in groups:
                mask = sb == g
                if mask.sum() < 2:
                    continue
                total_loss = total_loss + per_sample[mask].mean()
                n += 1

        if n == 0:
            return None, 0.0
        (total_loss / n).backward()
        return self.model.get_flat_grad().clone(), (total_loss / n).item()

    def _safety_gradient_disparity(self, xb, yb, sb, mem_batch=None):
        """Disparity safety gradient: g_worst_group − g_best_group.

        Single backward pass via loss = L_worst − L_best. Cancels the legitimate-
        feature agreement between groups, leaving only the spurious-feature conflict.
        Scales with actual disparity — near-zero when groups are similar.
        """
        self._last_safety_diag = None
        if mem_batch is not None:
            mx, my, ms = mem_batch
            xb = torch.cat([xb, mx])
            yb = torch.cat([yb, my])
            sb = torch.cat([sb, ms])

        groups = sb.unique()
        self.optimizer.zero_grad()
        # BatchNorm running statistics are frozen for the whole safety forward,
        # including when mem_batch is None (as train_task calls it), so the
        # safety pass adds no second momentum update per step relative to A-GEM.
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

    def _safety_gradient(self, xb, yb, sb, mem_batch=None):
        if self.use_topk_dro:
            return self._safety_gradient_topk(xb, yb, sb)
        if self.use_disparity:
            if self.disparity_mode == "conditional":
                return self._safety_gradient_conditional(xb, yb, sb)
            return self._safety_gradient_disparity(xb, yb, sb, mem_batch=mem_batch)
        return self._safety_gradient_average(xb, yb, sb, mem_batch=mem_batch)

    def _memory_gradient(self, mem_batch=None):
        if mem_batch is None:
            mem_batch = self.memory.get_random_batch(AGEM_MEM_BATCH_SIZE)
        if mem_batch is None:
            return None
        X_mem, y_mem, _ = mem_batch
        self.optimizer.zero_grad()
        with batchnorm_eval(self.model):
            nn.CrossEntropyLoss()(mask_logits(self.model(X_mem), self._seen_classes), y_mem).backward()
        return self.model.get_flat_grad().clone()

    def _project_dual(self, g_task, g_mem, g_safety, eps):
        """Cyclic projection onto the memory and safety half-spaces.

        g_safety is normalised so eps is scale-invariant:
          eps > 0 → the update must actively reduce disparity.
          eps = 0 → the update merely does not increase disparity.

        This is a finite feasibility procedure, not a nearest-point solver.
        """
        g = g_task.clone()
        g_safety_n = g_safety / (g_safety.norm() + 1e-8) if g_safety is not None else None
        for _ in range(self.proj_iters):
            if g_mem is not None:
                dot = (g * g_mem).sum()
                if dot < 0:
                    g = g - (dot / (g_mem.pow(2).sum() + 1e-8)) * g_mem
            if g_safety_n is not None:
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
        _n_mem_proj    = 0
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
                loss.backward()
                g_task = self.model.get_flat_grad().clone()
                batch_loss += loss.item()

                mem_batch = self.memory.get_random_batch(AGEM_MEM_BATCH_SIZE)
                g_mem = self._memory_gradient(mem_batch)
                g_safety, dro_loss = self._safety_gradient(xb, yb, sb, mem_batch=None)
                diag = getattr(self, "_last_safety_diag", None)
                if diag is not None:
                    _loss_gaps.append(diag["loss_gap"])
                    if "worst_group" in diag:
                        _worst_groups.append(diag["worst_group"])
                        _best_groups.append(diag["best_group"])

                if self.dynamic_eps:
                    self.current_eps = min(self.alpha, self.beta * dro_loss)
                else:
                    self.current_eps = self.alpha

                _n_steps += 1
                _dro_losses.append(dro_loss)
                _eps_vals.append(self.current_eps)
                if g_safety is not None:
                    g_sn = g_safety / (g_safety.norm() + 1e-8)
                    dot_s = float((g_task * g_sn).sum().item())
                    _dot_safety.append(dot_s)
                    if dot_s < self.current_eps:
                        _n_safety_proj += 1
                if g_mem is not None:
                    if float((g_task * g_mem).sum().item()) < 0:
                        _n_mem_proj += 1

                g_proj = self._project_dual(g_task, g_mem, g_safety, self.current_eps)
                g_clip = _clip_flat_grad(g_proj, MLP_GRAD_CLIP)
                self.model.set_flat_grad(g_clip)
                if _diag_sink is not None and _n_steps % 4 == 1:
                    _apply_step_with_diag(self.model, self.optimizer, g_clip,
                                          g_safety, g_mem, self.current_eps, _diag_sink)
                else:
                    self.optimizer.step()
                scheduler.step()
            epoch_losses.append(batch_loss / len(loader))
            if has_converged(epoch_losses):
                break

        self.proj_stats: dict = {
            "safety_proj_rate": _n_safety_proj / _n_steps if _n_steps else 0.0,
            "mem_proj_rate":    _n_mem_proj    / _n_steps if _n_steps else 0.0,
            "mean_dro_loss":    float(np.mean(_dro_losses)) if _dro_losses else 0.0,
            "mean_eps_t":       float(np.mean(_eps_vals))   if _eps_vals   else 0.0,
            "mean_dot_safety":  float(np.mean(_dot_safety)) if _dot_safety else 0.0,
            "n_steps":          _n_steps,
        }
        self.proj_stats.update(_diag_stats(_loss_gaps, _worst_groups, _best_groups))
        self.proj_stats.update(_realized_stats(_diag_sink))
