"""
baselines_neural.py — Neural continual-learning baselines.

  JointRetrain : Full-data reference. Resets and retrains on all tasks so far.
  Finetune     : Sequential fine-tuning. The catastrophic-forgetting lower bound.
  AGEM         : Averaged Gradient Episodic Memory (Chaudhry et al., ICLR 2019).
  DERpp        : Dark Experience Replay++ (Buzzega et al., NeurIPS 2020).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import (make_model, make_optimizer, make_scheduler, has_converged,
                   predict, mask_logits, _x_tensor, _train_batch_size,
                   batchnorm_eval)
from memory import EpisodicMemory
from config import (
    MLP_EPOCHS,
    MLP_GRAD_CLIP,
    DEVICE,
    AGEM_MEM_BATCH_SIZE,
    DERPP_ALPHA, DERPP_BETA,
)
import runctx


def _make_loader(dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(runctx.effective_seed())  # --seed varies batch order
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g)


def _apply_step_with_diag(model, optimizer, g_applied, g_safety, g_mem, eps, sink):
    """Audit the optimizer's norm-matched effective descent direction.

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


def _clip_flat_grad(g: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = g.norm()
    if norm > max_norm:
        g = g * (max_norm / (norm + 1e-8))
    return g


class JointRetrain:
    """Full-data reference: resets and retrains on all tasks seen so far.

    Not a CL method — stores the full history and provides an ERM reference
    under the same architecture.
    """
    def __init__(self, input_dim: int, device: str = DEVICE, bert_mode: bool = False,
                 bert_lr: float = None):
        self.input_dim = input_dim
        self.device = device
        self.bert_mode = bert_mode
        self.bert_lr = bert_lr
        self.model = make_model(input_dim, bert_mode).to(device)
        self.optimizer = make_optimizer(self.model, bert_lr=bert_lr)
        self.criterion = nn.CrossEntropyLoss()
        self.all_X: list[np.ndarray] = []
        self.all_y: list[np.ndarray] = []

    def train_task(self, X_train: np.ndarray, y_train: np.ndarray,
                   epochs: int = MLP_EPOCHS, seen_classes: list[int] | None = None):
        self.all_X.append(X_train)
        self.all_y.append(y_train)
        X_all = np.concatenate(self.all_X)
        y_all = np.concatenate(self.all_y)

        self.model = make_model(self.input_dim, self.bert_mode).to(self.device)
        self.optimizer = make_optimizer(self.model, bert_lr=self.bert_lr)

        X_t = _x_tensor(X_all)
        y_t = torch.tensor(y_all, dtype=torch.long)
        loader = _make_loader(TensorDataset(X_t, y_t), batch_size=_train_batch_size(self.bert_mode))

        scheduler = make_scheduler(self.optimizer, epochs, steps_per_epoch=len(loader))
        self.model.train()
        epoch_losses = []
        for _ in range(epochs):
            batch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(mask_logits(self.model(xb), seen_classes), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), MLP_GRAD_CLIP)
                self.optimizer.step()
                scheduler.step()
                batch_loss += loss.item()
            epoch_losses.append(batch_loss / len(loader))
            if has_converged(epoch_losses):
                break

    def predict(self, X: np.ndarray, seen_classes: list[int] | None = None) -> np.ndarray:
        return predict(self.model, X, self.device, seen_classes=seen_classes)



class Finetune:
    """Naive sequential fine-tuning — the catastrophic-forgetting lower bound."""

    def __init__(self, input_dim: int, device: str = DEVICE, bert_mode: bool = False,
                 bert_lr: float = None):
        self.input_dim = input_dim
        self.device = device
        self.bert_mode = bert_mode
        self.bert_lr = bert_lr
        self.model = make_model(input_dim, bert_mode).to(device)
        self.optimizer = make_optimizer(self.model, bert_lr=bert_lr)
        self.criterion = nn.CrossEntropyLoss()

    def train_task(self, X_train: np.ndarray, y_train: np.ndarray,
                   epochs: int = MLP_EPOCHS, seen_classes: list[int] | None = None):
        X_t = _x_tensor(X_train)
        y_t = torch.tensor(y_train, dtype=torch.long)
        loader = _make_loader(TensorDataset(X_t, y_t), batch_size=_train_batch_size(self.bert_mode))

        scheduler = make_scheduler(self.optimizer, epochs, steps_per_epoch=len(loader))
        self.model.train()
        epoch_losses = []
        for _ in range(epochs):
            batch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(mask_logits(self.model(xb), seen_classes), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), MLP_GRAD_CLIP)
                self.optimizer.step()
                scheduler.step()
                batch_loss += loss.item()
            epoch_losses.append(batch_loss / len(loader))
            if has_converged(epoch_losses):
                break

    def predict(self, X: np.ndarray, seen_classes: list[int] | None = None) -> np.ndarray:
        return predict(self.model, X, self.device, seen_classes=seen_classes)



class AGEM:
    """Averaged Gradient Episodic Memory (Chaudhry et al., ICLR 2019).

    Projects the task gradient g_t so it doesn't increase the loss on memory:

        if dot(g_t, g_mem) >= 0: use g_t unchanged
        else: g_t -= (dot(g_t, g_mem) / ||g_mem||^2) * g_mem

    Also defines the interface that FAIRGEM subclasses.
    """
    def __init__(self, input_dim: int, device: str = DEVICE, bert_mode: bool = False,
                 bert_lr: float = None):
        self.input_dim = input_dim
        self.device = device
        self.bert_mode = bert_mode
        self.bert_lr = bert_lr
        self.model = make_model(input_dim, bert_mode).to(device)
        self.optimizer = make_optimizer(self.model, bert_lr=bert_lr)
        self.criterion = nn.CrossEntropyLoss()
        self.memory = EpisodicMemory(device=device)

    def remember(self, X: np.ndarray, y: np.ndarray, s: np.ndarray):
        self.memory.add_task(X, y, s)

    def _project_agem(self, g_task: torch.Tensor, g_mem: torch.Tensor) -> torch.Tensor:
        dot = (g_task * g_mem).sum()
        if dot < 0:
            g_task = g_task - (dot / (g_mem.pow(2).sum() + 1e-8)) * g_mem
        return g_task

    def train_task(self, X_train: np.ndarray, y_train: np.ndarray,
                   s_train: np.ndarray = None, epochs: int = MLP_EPOCHS,
                   seen_classes: list[int] | None = None):
        if s_train is None:
            s_train = np.zeros(len(y_train), dtype=np.int64)

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
        for _ in range(epochs):
            batch_loss = 0.0
            for xb, yb, _ in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                self.optimizer.zero_grad()
                loss = self.criterion(mask_logits(self.model(xb), seen_classes), yb)
                loss.backward()
                g_task = self.model.get_flat_grad().clone()
                batch_loss += loss.item()

                mem = self.memory.get_random_batch(AGEM_MEM_BATCH_SIZE)
                if mem is not None:
                    X_mem, y_mem, _ = mem
                    self.optimizer.zero_grad()
                    with batchnorm_eval(self.model):
                        self.criterion(mask_logits(self.model(X_mem), seen_classes), y_mem).backward()
                    g_mem = self.model.get_flat_grad().clone()
                    g_task = self._project_agem(g_task, g_mem)

                self.model.set_flat_grad(_clip_flat_grad(g_task, MLP_GRAD_CLIP))
                self.optimizer.step()
                scheduler.step()
            epoch_losses.append(batch_loss / len(loader))
            if has_converged(epoch_losses):
                break

    def predict(self, X: np.ndarray, seen_classes: list[int] | None = None) -> np.ndarray:
        return predict(self.model, X, self.device, seen_classes=seen_classes)



class DERpp:
    """Dark Experience Replay++ (Buzzega et al., NeurIPS 2020).

    Per-step objective:
        L = CE(f(x_t), y_t)
          + alpha * MSE(f(x_mem), z_mem)   [distill stored logits]
          + beta  * CE(f(x_mem), y_mem)    [reinforce stored labels]

    CL-only ablation baseline for FairDERpp: same memory, no fairness projection.
    """

    def __init__(
        self,
        input_dim: int,
        device: str = DEVICE,
        alpha: float = DERPP_ALPHA,
        beta: float = DERPP_BETA,
        bert_mode: bool = False,
        bert_lr: float = None,
    ):
        self.device    = device
        self.alpha     = alpha
        self.beta      = beta
        self.bert_mode = bert_mode
        self.bert_lr   = bert_lr

        self.model     = make_model(input_dim, bert_mode).to(device)
        self.optimizer = make_optimizer(self.model, bert_lr=bert_lr)
        self.memory    = EpisodicMemory(device=device)

    def train_task(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        s_train: np.ndarray = None,
        epochs: int = MLP_EPOCHS,
        seen_classes: list[int] | None = None,
    ):
        X_t = _x_tensor(X_train)
        y_t = torch.tensor(y_train, dtype=torch.long)
        loader = _make_loader(
            TensorDataset(X_t, y_t),
            batch_size=_train_batch_size(self.bert_mode),
        )

        scheduler    = make_scheduler(self.optimizer, epochs, steps_per_epoch=len(loader))
        self.model.train()
        epoch_losses = []

        for _ in range(epochs):
            batch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                self.optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(mask_logits(self.model(xb), seen_classes), yb)

                # Two independent memory batches, as in Buzzega et al.: one for
                # logit distillation, one for label replay (a single shared
                # batch would correlate the two terms). Fair-DER++ draws the
                # same way.
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
                    loss = (
                        loss
                        + self.alpha * mse
                        + self.beta  * nn.CrossEntropyLoss()(mask_logits(logits_r, seen_classes), yr)
                    )

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), MLP_GRAD_CLIP)
                self.optimizer.step()
                scheduler.step()
                batch_loss += loss.item()

            epoch_losses.append(batch_loss / len(loader))
            if has_converged(epoch_losses):
                break

    def remember(self, X: np.ndarray, y: np.ndarray, s: np.ndarray):
        """Compute logits on X and store (X, y, s, logits). Call right after train_task."""
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
