"""
fairderpp.py — DER++ with a fairness safety projection.

DER++ handles continual learning via logit distillation replay. On top of that,
a single fairness projection keeps each update from increasing the group disparity.

Unlike FAIR-GEM's cyclic dual projection, only one constraint is needed here
because DER++ already handles memory retention through the loss terms.

Ablation flags:
  use_disparity=False - average-group gradient as the safety direction, which
                        removes the disparity signal and degenerates the method
                        towards plain DER++. Mirrors FAIRGEM's flag of the same
                        name.
  dynamic_eps=False   - static epsilon = eps_alpha.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from memory import EpisodicMemory
from model import (MLP, make_optimizer, make_scheduler, has_converged, predict,
                   batchnorm_eval)
from baselines_neural import (_clip_flat_grad, _apply_step_with_diag,
                              _diag_sink_or_none, _realized_stats)
from config import (
    MLP_EPOCHS, MLP_BATCH_SIZE, MLP_GRAD_CLIP, DEVICE,
    FAIRDERPP_ALPHA, FAIRDERPP_BETA,
    FAIRDERPP_EPS_ALPHA, FAIRDERPP_EPS_BETA,
    FAIRDERPP_MEM_SIZE, FAIRDERPP_MEM_BATCH,
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


class FairDERpp:
    """
    Memory stores (X, y, s, logits) tuples. At each step the DER++ replay loss
    acts as the continual-learning constraint (no gradient projection needed for
    memory). A single fairness projection is applied on top to keep the update
    from increasing the group-loss disparity surrogate.
    """

    def __init__(
        self,
        input_dim: int,
        device: str = DEVICE,
        alpha: float = FAIRDERPP_ALPHA,
        beta: float = FAIRDERPP_BETA,
        eps_alpha: float = FAIRDERPP_EPS_ALPHA,
        eps_beta: float = FAIRDERPP_EPS_BETA,
        mem_size: int = FAIRDERPP_MEM_SIZE,
        mem_batch: int = FAIRDERPP_MEM_BATCH,
        dynamic_eps: bool = True,
        use_disparity: bool = True,
        opt: str = "adamw",
        opt_lr: float | None = None,
        disparity_mode: str = "unconditional",
    ):
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.eps_alpha = eps_alpha
        self.eps_beta = eps_beta
        self.current_eps = eps_alpha
        self.dynamic_eps = dynamic_eps
        self.use_disparity = use_disparity
        self.disparity_mode = validate_disparity_mode(disparity_mode)
        self.mem_batch_size = int(mem_batch)
        self.model = MLP(input_dim).to(device)
        self.optimizer = make_optimizer(self.model)
        # opt="sgd" uses plain SGD (no momentum, no weight decay): the realized
        # step is then exactly -lr * g_proj, so the projected half-space
        # guarantee holds for the parameter update itself (the AdamW default
        # rotates it - see realized_* diagnostics). Ablation arms only.
        if opt == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=opt_lr if opt_lr is not None else 1e-3)
        self.memory = EpisodicMemory(size_per_task=int(mem_size), device=device)


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

    def _safety_gradient_disparity(self, xb, yb, sb):
        """
        Disparity safety gradient: g_worst_group − g_best_group.
        """
        self._last_safety_diag = None
        groups = sb.unique()
        self.optimizer.zero_grad()
        # BatchNorm in eval mode for the safety pass: only the main DER++ forward
        # should update running statistics, so Fair-DER++ takes the same one
        # momentum update per step as DER++.
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

    def _safety_gradient_average(self, xb, yb, sb):
        """
        Average-group gradient (ablation): the safety direction becomes the mean
        of the per-group losses instead of worst - best. Everything else — the
        DER++ loss, the projection, the eps schedule — is untouched, so the arm
        isolates the disparity signal itself rather than how hard it is applied.

        Mirrors FAIRGEM._safety_gradient_average, so fair_gem_avg_grad and
        fair_derpp_avg_grad measure the same thing.
        """
        self._last_safety_diag = None
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
        # As in fairgem: the eps schedule expects a worst-best disparity gap
        # (~0.0-0.3), not a raw mean loss (~0.7, which would pin eps at
        # eps_alpha on every step), so the ablation changes only the gradient
        # estimator.
        with torch.no_grad():
            gls = [per_sample[sb == g].mean().item()
                   for g in groups if (sb == g).sum() >= 2]
        gap = (max(gls) - min(gls)) if len(gls) >= 2 else 0.0
        return self.model.get_flat_grad().clone(), gap

    def _safety_gradient(self, xb, yb, sb):
        if self.use_disparity:
            if self.disparity_mode == "conditional":
                return self._safety_gradient_conditional(xb, yb, sb)
            return self._safety_gradient_disparity(xb, yb, sb)
        return self._safety_gradient_average(xb, yb, sb)

    def _project_fairness(self, g, g_safety, eps):
        """Project g onto the half-space dot(g, g_safety_n) >= eps."""
        if g_safety is None:
            return g
        g_safety_n = g_safety / (g_safety.norm() + 1e-8)
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

                # DER++ combined loss
                self.optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(self.model(xb), yb)
                # Two independent memory batches, matching Buzzega et al. and the
                # DER++ baseline.
                mem_distill = self.memory.get_random_batch_with_logits(self.mem_batch_size)
                mem_replay = self.memory.get_random_batch_with_logits(self.mem_batch_size)
                if mem_distill is not None:
                    xd, _yd, _sd, zd = mem_distill
                    xr, yr, _sr, _zr = mem_replay
                    with batchnorm_eval(self.model):
                        logits_d = self.model(xd)
                        logits_r = self.model(xr)
                    target = zd.detach()
                    loss = (loss
                            + self.alpha * F.mse_loss(logits_d, target)
                            + self.beta  * nn.CrossEntropyLoss()(logits_r, yr))
                loss.backward()
                g_der = self.model.get_flat_grad().clone()
                batch_loss += loss.item()

                # Fairness projection
                g_safety, dro_loss = self._safety_gradient(xb, yb, sb)
                diag = getattr(self, "_last_safety_diag", None)
                if diag is not None:
                    _loss_gaps.append(diag["loss_gap"])
                    if "worst_group" in diag:
                        _worst_groups.append(diag["worst_group"])
                        _best_groups.append(diag["best_group"])
                if self.dynamic_eps:
                    self.current_eps = min(self.eps_alpha, self.eps_beta * dro_loss)
                else:
                    self.current_eps = self.eps_alpha

                _n_steps += 1
                _dro_losses.append(dro_loss)
                _eps_vals.append(self.current_eps)
                if g_safety is not None:
                    g_sn  = g_safety / (g_safety.norm() + 1e-8)
                    dot_s = float((g_der * g_sn).sum().item())
                    _dot_safety.append(dot_s)
                    if dot_s < self.current_eps:
                        _n_safety_proj += 1

                # Clip first, then project, so clipping cannot shrink the
                # safety margin; see the note in fairgem.py.
                g_clipped = _clip_flat_grad(g_der, MLP_GRAD_CLIP)
                g_proj = self._project_fairness(g_clipped, g_safety, self.current_eps)
                self.model.set_flat_grad(g_proj)
                if _diag_sink is not None and _n_steps % 4 == 1:
                    _apply_step_with_diag(self.model, self.optimizer, g_proj,
                                          g_safety, None, self.current_eps, _diag_sink)
                else:
                    self.optimizer.step()
            scheduler.step()
            epoch_losses.append(batch_loss / len(loader))
            if has_converged(epoch_losses):
                break

        self.proj_stats = {
            "safety_proj_rate": _n_safety_proj / _n_steps if _n_steps else 0.0,
            "mean_dro_loss":    float(np.mean(_dro_losses)) if _dro_losses else 0.0,
            "mean_eps_t":       float(np.mean(_eps_vals))   if _eps_vals   else 0.0,
            "mean_dot_safety":  float(np.mean(_dot_safety)) if _dot_safety else 0.0,
            "n_steps":          _n_steps,
        }
        self.proj_stats.update(_diag_stats(_loss_gaps, _worst_groups, _best_groups))
        self.proj_stats.update(_realized_stats(_diag_sink))

    def remember(self, X, y, s):
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
        self.model.train()
        self.memory.add_task_with_logits(X, y, s, logits)

    def predict(self, X):
        return predict(self.model, X, self.device)
