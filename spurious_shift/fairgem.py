"""
fairgem.py - fairness-aware continual learning via dual gradient projection.

Extends A-GEM with:
  1. A safety gradient g_safety that points in the direction of increasing disparity
     (L_worst - L_best, so projecting against it reduces the group gap).
  2. A dynamic safety margin: so the projection is quiet when groups are already similar
     and fires hard when they diverge.
  3. Cyclic dual projection: alternates memory and safety half-space corrections.

Ablation flags:
  use_topk_dro=True   - top-k DRO on worst-group samples from the current batch only.
                         Noisier: worst-group identification can flip early in training.
  use_disparity=False - average-group gradient (degenerates to A-GEM).
  dynamic_eps=False   - static epsilon = alpha.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import make_scheduler, has_converged, batchnorm_eval
from baselines_neural import (AGEM, _clip_flat_grad, _apply_step_with_diag,
                              _diag_sink_or_none, _realized_stats)
from config import (
    MLP_EPOCHS, MLP_BATCH_SIZE, DEVICE, MLP_GRAD_CLIP,
    FAIRGEM_TOP_K, FAIRGEM_ALPHA, FAIRGEM_BETA,
    FAIRGEM_MEM_SIZE, FAIRGEM_MEM_BATCH,
)
from safety import conditional_disparity, validate_disparity_mode


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

    def __init__(self, input_dim, device=DEVICE, top_k=FAIRGEM_TOP_K,
                 alpha=FAIRGEM_ALPHA, beta=FAIRGEM_BETA,
                 use_topk_dro=False, use_disparity=True, dynamic_eps=True,
                 proj_iters=15,
                 mem_size=FAIRGEM_MEM_SIZE, mem_batch=FAIRGEM_MEM_BATCH,
                 opt="adamw", opt_lr=None,
                 disparity_mode="unconditional"):
        super().__init__(input_dim, device, mem_size=mem_size, mem_batch=mem_batch)
        # opt="sgd" uses plain SGD (no momentum, no weight decay): the realized
        # step is then exactly -lr * g_proj, so the projected half-space
        # guarantee holds for the parameter update itself (the AdamW default
        # rotates it - see realized_* diagnostics). Ablation arms only.
        if opt == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=opt_lr if opt_lr is not None else 1e-3)
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
        """Top-k DRO on worst-group samples from current batch (ablation)."""
        criterion = nn.CrossEntropyLoss(reduction="none")
        self._last_safety_diag = None
        # Every forward here is auxiliary, so none may touch BatchNorm's running
        # statistics. torch.no_grad() alone does not prevent that (it stops
        # gradient tracking, not the momentum update), and the group-masked
        # forwards would otherwise normalise over a single group.
        with batchnorm_eval(self.model):
            group_losses = {}
            worst_loss, worst_group = -1.0, None
            with torch.no_grad():
                for g in sb.unique():
                    mask = sb == g
                    if mask.sum() < 2:
                        continue
                    loss = criterion(self.model(xb[mask]), yb[mask]).mean().item()
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

            mask = sb == worst_group
            xg, yg = xb[mask], yb[mask]
            with torch.no_grad():
                per_sample = criterion(self.model(xg), yg)
            k = min(self.top_k, len(per_sample))
            top_idx = per_sample.topk(k).indices
            self.optimizer.zero_grad()
            dro_loss = nn.CrossEntropyLoss()(self.model(xg[top_idx]), yg[top_idx])
        dro_loss.backward()
        return self.model.get_flat_grad().clone(), dro_loss.item()

    def _safety_gradient_average(self, xb, yb, sb, mem_batch=None):
        """Average-group gradient (degenerates to A-GEM when used as the safety direction)."""
        self._last_safety_diag = None
        if mem_batch is not None:
            mx, my, ms = mem_batch
            xb, yb, sb = torch.cat([xb, mx]), torch.cat([yb, my]), torch.cat([sb, ms])

        groups = sb.unique()
        self.optimizer.zero_grad()
        with batchnorm_eval(self.model):
            logits = self.model(xb)
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
        # The eps schedule expects a worst-best disparity gap (~0.0-0.3), not a
        # raw mean loss (~0.7, which pins eps at alpha on every step). Feeding
        # the gap keeps this ablation a change of gradient estimator only.
        with torch.no_grad():
            gls = [per_sample[sb == g].mean().item() for g in groups if (sb == g).sum() >= 2]
        gap = (max(gls) - min(gls)) if len(gls) >= 2 else 0.0
        return self.model.get_flat_grad().clone(), gap

    def _safety_gradient_disparity(self, xb, yb, sb, mem_batch=None):
        """
        Disparity safety gradient: g_worst_group - g_best_group.
        """
        self._last_safety_diag = None
        if mem_batch is not None:
            mx, my, ms = mem_batch
            xb, yb, sb = torch.cat([xb, mx]), torch.cat([yb, my]), torch.cat([sb, ms])

        groups = sb.unique()
        self.optimizer.zero_grad()
        # BatchNorm stays in eval mode for the whole safety forward, so FAIR-GEM
        # takes the same one running-statistics update per step as A-GEM.
        with batchnorm_eval(self.model):
            logits = self.model(xb)
            per_sample = nn.CrossEntropyLoss(reduction="none")(logits, yb)

            group_losses = {}
            for g in groups:
                mask = sb == g
                if mask.sum() >= 2:
                    group_losses[g.item()] = per_sample[mask].mean()

        if len(group_losses) < 2:
            # One group in the batch means disparity is undefined: skip the
            # safety constraint this step.
            return None, 0.0

        sorted_gs = sorted(group_losses, key=lambda g: group_losses[g].item())
        best_g, worst_g = sorted_gs[0], sorted_gs[-1]
        disparity_loss = group_losses[worst_g] - group_losses[best_g]
        self._last_safety_diag = {
            "best_group": int(best_g),
            "worst_group": int(worst_g),
            "loss_gap": float(disparity_loss.detach().item()),
        }
        disparity_loss.backward()
        return self.model.get_flat_grad().clone(), disparity_loss.item()


    def _safety_gradient_conditional(self, xb, yb, sb):
        """Gradient of the mean estimable within-label group-loss gap."""
        self._last_safety_diag = None
        self.optimizer.zero_grad()
        with batchnorm_eval(self.model):
            logits = self.model(xb)
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
            mem_batch = self.memory.get_random_batch(self.mem_batch_size)
        if mem_batch is None:
            return None
        X_mem, y_mem, _ = mem_batch
        self.optimizer.zero_grad()
        with batchnorm_eval(self.model):
            nn.CrossEntropyLoss()(self.model(X_mem), y_mem).backward()
        return self.model.get_flat_grad().clone()

    def _project_dual(self, g_task, g_mem, g_safety, eps):
        """
        Cyclic projections onto the memory and safety half-spaces.

        This feasibility loop does not use Dykstra correction variables and
        therefore does not claim the nearest point in the intersection.
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

    def train_task(self, X_train, y_train, s_train, epochs=MLP_EPOCHS):
        loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.long),
                          torch.tensor(s_train, dtype=torch.long)),
            batch_size=MLP_BATCH_SIZE, shuffle=True,
        )
        scheduler = make_scheduler(self.optimizer, epochs)
        self.model.train()
        epoch_losses = []

        _diag_sink = _diag_sink_or_none()
        _n_steps = 0
        _n_safety_proj = 0
        _n_mem_proj = 0
        _dro_losses = []
        _eps_vals = []
        _dot_safety = []
        _loss_gaps = []
        _worst_groups = []
        _best_groups = []

        for _ in range(epochs):
            batch_loss = 0.0
            for xb, yb, sb in loader:
                xb, yb, sb = xb.to(self.device), yb.to(self.device), sb.to(self.device)

                self.optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(self.model(xb), yb)
                loss.backward()
                g_task = self.model.get_flat_grad().clone()
                batch_loss += loss.item()

                mem_batch = self.memory.get_random_batch(self.mem_batch_size)
                g_mem = self._memory_gradient(mem_batch)
                # Safety gradient on current batch only — memory from past fair tasks
                # would dilute the disparity signal and weaken the projection.
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

                # Track projection activation stats
                _n_steps += 1
                _dro_losses.append(dro_loss)
                _eps_vals.append(self.current_eps)
                if g_safety is not None:
                    g_sn = g_safety / (g_safety.norm() + 1e-8)
                    dot_s = float((g_task * g_sn).sum().item())
                    _dot_safety.append(dot_s)
                    if dot_s < self.current_eps:
                        _n_safety_proj += 1
                if g_mem is not None and float((g_task * g_mem).sum().item()) < 0:
                    _n_mem_proj += 1

                # Clip first, then project. Clipping afterwards rescales the
                # gradient by c < 1, which turns the satisfied constraint
                # <g, g_safe_hat> = eps into c*eps < eps, shrinking the margin
                # when the gradient is large. The projection adds at most eps
                # along a unit vector, so the final norm is bounded by
                # MLP_GRAD_CLIP + eps.
                g_clipped = _clip_flat_grad(g_task, MLP_GRAD_CLIP)
                g_proj = self._project_dual(g_clipped, g_mem, g_safety, self.current_eps)
                self.model.set_flat_grad(g_proj)
                if _diag_sink is not None and _n_steps % 4 == 1:
                    _apply_step_with_diag(self.model, self.optimizer, g_proj,
                                          g_safety, g_mem, self.current_eps, _diag_sink)
                else:
                    self.optimizer.step()
            scheduler.step()
            epoch_losses.append(batch_loss / len(loader))
            if has_converged(epoch_losses):
                break

        self.proj_stats = {
            "safety_proj_rate": _n_safety_proj / _n_steps if _n_steps else 0.0,
            "mem_proj_rate":    _n_mem_proj    / _n_steps if _n_steps else 0.0,
            "mean_dro_loss":    float(np.mean(_dro_losses)) if _dro_losses else 0.0,
            "mean_eps_t":       float(np.mean(_eps_vals))   if _eps_vals   else 0.0,
            "mean_dot_safety":  float(np.mean(_dot_safety)) if _dot_safety else 0.0,
            "n_steps":          _n_steps,
        }
        self.proj_stats.update(_diag_stats(_loss_gaps, _worst_groups, _best_groups))
        self.proj_stats.update(_realized_stats(_diag_sink))
