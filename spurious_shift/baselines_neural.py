"""
baselines_neural.py — Neural continual-learning baselines.

  JointRetrain: Full-data reference. Resets and retrains from scratch on all tasks seen so far.
  Finetune: Sequential fine-tuning with no forgetting prevention.
  AGEM: Averaged Gradient Episodic Memory.
  DERpp: Dark Experience Replay++.

All methods share the same MLP backbone from model.py.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import (MLP, make_optimizer, make_scheduler, has_converged, predict,
                   batchnorm_eval)
from memory import EpisodicMemory
from config import (
    MLP_EPOCHS, MLP_BATCH_SIZE, MLP_GRAD_CLIP, DEVICE,
    MEMORY_SIZE_PER_TASK, AGEM_MEM_BATCH_SIZE,
    DERPP_ALPHA, DERPP_BETA, DERPP_MEM_SIZE, DERPP_MEM_BATCH,
)


def _apply_step_with_diag(model, optimizer, g_applied, g_safety, g_mem, eps, sink):
    """Record the optimizer's norm-matched effective descent direction.

    With ``CLFAIR_STEP_DIAG=1``, records cos(-dtheta, g_applied). The safety
    check rescales ``-dtheta`` to ``||g_applied||`` before applying the original
    margin; the memory check is scale-invariant and uses its sign directly.
    """
    def _flat_params():
        return torch.cat([p.data.view(-1) for p in model.parameters()])
    theta0 = _flat_params().clone()
    optimizer.step()
    with torch.no_grad():
        d = _flat_params() - theta0
        dn = d.norm() + 1e-12
        gn = g_applied.norm() + 1e-12
        sink["cos_update"].append(float((-d * g_applied).sum() / (dn * gn)))
        if g_safety is not None:
            g_sn = g_safety / (g_safety.norm() + 1e-8)
            sink["safety_ok"].append(float((-d * g_sn).sum() / dn) >= float(eps) / float(gn))
        if g_mem is not None:
            sink["mem_ok"].append(float((-d * g_mem).sum()) >= 0.0)


def _diag_sink_or_none():
    import os
    if os.environ.get("CLFAIR_STEP_DIAG") == "1":
        return {"cos_update": [], "safety_ok": [], "mem_ok": []}
    return None


def _realized_stats(sink):
    import numpy as _np
    if sink is None:
        return {}
    return {
        "realized_cos_update":  float(_np.mean(sink["cos_update"])) if sink["cos_update"] else float("nan"),
        "realized_safety_rate": float(_np.mean(sink["safety_ok"]))  if sink["safety_ok"]  else float("nan"),
        "realized_mem_rate":    float(_np.mean(sink["mem_ok"]))     if sink["mem_ok"]     else float("nan"),
        "n_diag_steps":         len(sink["cos_update"]),
    }


def _clip_flat_grad(g, max_norm):
    norm = g.norm()
    if norm > max_norm:
        g = g * (max_norm / (norm + 1e-8))
    return g


class JointRetrain:
    """
    JointRetrain implementation: stores the full history and replays everything each task.
    """
    def __init__(self, input_dim, device=DEVICE):
        self.input_dim = input_dim
        self.device = device
        self.model = MLP(input_dim).to(device)
        self.optimizer = make_optimizer(self.model)
        self.criterion = nn.CrossEntropyLoss()
        self.all_X = []
        self.all_y = []

    def train_task(self, X_train, y_train, epochs=MLP_EPOCHS):
        self.all_X.append(X_train)
        self.all_y.append(y_train)
        X_all, y_all = np.concatenate(self.all_X), np.concatenate(self.all_y)

        self.model = MLP(self.input_dim).to(self.device)
        self.optimizer = make_optimizer(self.model)

        loader = DataLoader(TensorDataset(torch.tensor(X_all, dtype=torch.float32),
                                          torch.tensor(y_all, dtype=torch.long)),
                               batch_size=MLP_BATCH_SIZE, shuffle=True)
        scheduler = make_scheduler(self.optimizer, epochs)
        self.model.train()
        epoch_losses = []
        for _ in range(epochs):
            batch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), MLP_GRAD_CLIP)
                self.optimizer.step()
                batch_loss += loss.item()
            scheduler.step()
            epoch_losses.append(batch_loss / len(loader))
            if has_converged(epoch_losses):
                break

    def predict(self, X):
        return predict(self.model, X, self.device)



class Finetune:
    """Sequential fine-tuning with no forgetting prevention (catastrophic-forgetting lower bound)."""

    def __init__(self, input_dim, device=DEVICE):
        self.input_dim = input_dim
        self.device    = device
        self.model     = MLP(input_dim).to(device)
        self.optimizer = make_optimizer(self.model)
        self.criterion = nn.CrossEntropyLoss()

    def train_task(self, X_train, y_train, epochs=MLP_EPOCHS):
        loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                          torch.tensor(y_train, dtype=torch.long)),
                               batch_size=MLP_BATCH_SIZE, shuffle=True)
        scheduler = make_scheduler(self.optimizer, epochs)
        self.model.train()
        epoch_losses = []
        for _ in range(epochs):
            batch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), MLP_GRAD_CLIP)
                self.optimizer.step()
                batch_loss += loss.item()
            scheduler.step()
            epoch_losses.append(batch_loss / len(loader))
            if has_converged(epoch_losses):
                break

    def predict(self, X):
        return predict(self.model, X, self.device)



class AGEM:
    """
    Averaged Gradient Episodic Memory implementation

    Shared with FAIRGEM subclasses.
    """
    def __init__(self, input_dim, device=DEVICE,
                 mem_size=MEMORY_SIZE_PER_TASK, mem_batch=AGEM_MEM_BATCH_SIZE):
        self.input_dim = input_dim
        self.device = device
        self.mem_batch_size = int(mem_batch)
        self.model = MLP(input_dim).to(device)
        self.optimizer = make_optimizer(self.model)
        self.criterion = nn.CrossEntropyLoss()
        self.memory = EpisodicMemory(size_per_task=int(mem_size), device=device)

    def remember(self, X, y, s):
        self.memory.add_task(X, y, s)

    def _project_agem(self, g_task, g_mem):
        dot = (g_task * g_mem).sum()
        if dot < 0:
            g_task = g_task - (dot / (g_mem.pow(2).sum() + 1e-8)) * g_mem
        return g_task

    def train_task(self, X_train, y_train, s_train=None, epochs=MLP_EPOCHS):
        if s_train is None:
            s_train = np.zeros(len(y_train), dtype=np.int64)

        loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.long),
                          torch.tensor(s_train, dtype=torch.long)),
            batch_size=MLP_BATCH_SIZE, shuffle=True,
        )
        scheduler = make_scheduler(self.optimizer, epochs)
        self.model.train()
        epoch_losses = []
        for _ in range(epochs):
            batch_loss = 0.0
            for xb, yb, _ in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                self.optimizer.zero_grad()
                loss = self.criterion(self.model(xb), yb)
                loss.backward()
                g_task = self.model.get_flat_grad().clone()
                batch_loss += loss.item()

                mem = self.memory.get_random_batch(self.mem_batch_size)
                if mem is not None:
                    X_mem, y_mem, _ = mem
                    self.optimizer.zero_grad()
                    with batchnorm_eval(self.model):
                        self.criterion(self.model(X_mem), y_mem).backward()
                    g_mem = self.model.get_flat_grad().clone()
                    g_task = self._project_agem(g_task, g_mem)

                self.model.set_flat_grad(_clip_flat_grad(g_task, MLP_GRAD_CLIP))
                self.optimizer.step()
            scheduler.step()
            epoch_losses.append(batch_loss / len(loader))
            if has_converged(epoch_losses):
                break

    def predict(self, X):
        return predict(self.model, X, self.device)



class DERpp:
    """
    Dark Experience Replay++ implementation.
    """
    def __init__(self, input_dim, device=DEVICE,
                 alpha=DERPP_ALPHA, beta=DERPP_BETA,
                 mem_size=DERPP_MEM_SIZE, mem_batch=DERPP_MEM_BATCH):
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.mem_batch_size = int(mem_batch)
        self.model = MLP(input_dim).to(device)
        self.optimizer = make_optimizer(self.model)
        self.memory = EpisodicMemory(size_per_task=int(mem_size), device=device)

    def train_task(self, X_train, y_train, s_train=None, epochs=MLP_EPOCHS):
        loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                          torch.tensor(y_train, dtype=torch.long)),
                            batch_size=MLP_BATCH_SIZE, shuffle=True)
        scheduler = make_scheduler(self.optimizer, epochs)
        self.model.train()
        epoch_losses = []
        for _ in range(epochs):
            batch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(self.model(xb), yb)

                # Buzzega et al. draw two *independent* memory batches — one for
                # the logit-distillation term, one for the label-replay term.
                mem_distill = self.memory.get_random_batch_with_logits(self.mem_batch_size)
                mem_replay = self.memory.get_random_batch_with_logits(self.mem_batch_size)
                if mem_distill is not None:
                    xd, _yd, _sd, zd = mem_distill
                    xr, yr, _sr, _zr = mem_replay
                    with batchnorm_eval(self.model):
                        logits_d = self.model(xd)
                        logits_r = self.model(xr)
                    loss = (loss
                            + self.alpha * F.mse_loss(logits_d, zd.detach())
                            + self.beta  * nn.CrossEntropyLoss()(logits_r, yr))

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), MLP_GRAD_CLIP)
                self.optimizer.step()
                batch_loss += loss.item()
            scheduler.step()
            epoch_losses.append(batch_loss / len(loader))
            if has_converged(epoch_losses):
                break

    def remember(self, X, y, s):
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
        self.model.train()
        self.memory.add_task_with_logits(X, y, s, logits)

    def predict(self, X):
        return predict(self.model, X, self.device)
