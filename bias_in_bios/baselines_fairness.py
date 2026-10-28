"""
baselines_fairness.py — Fairness-aware baselines for sequential learning.

  GroupDRO: Distributionally Robust Optimization over demographic groups
            with exponentiated group weight updates (Sagawa et al., ICLR 2020).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import make_model, make_optimizer, make_scheduler, has_converged, predict, mask_logits, _x_tensor, _train_batch_size
from config import (
    MLP_EPOCHS, MLP_GRAD_CLIP,
    DRO_ETA, DRO_ADJ,
    DEVICE,
)
import runctx


def _make_loader(dataset, batch_size, shuffle=True):
    g = torch.Generator(); g.manual_seed(runctx.effective_seed())  # --seed varies batch order
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g)


class GroupDRO:
    """Group DRO (Sagawa et al., ICLR 2020).

    Per step:
      q_g <- q_g * exp(eta * L_g)   [exponentiated gradient]
      q   <- normalize(q)
      Loss = sum_g q_g * (L_g + adj/sqrt(n_g))

    Retrained fresh each task for tabular (fairness-only baseline).
    Continual in bert_mode to match FaIRL's sequential BIOS setup.
    """
    def __init__(self, input_dim: int, eta: float = DRO_ETA,
                 adj: float = DRO_ADJ, device: str = DEVICE, bert_mode: bool = False,
                 bert_lr: float = None):
        self.input_dim = input_dim
        self.eta = eta
        self.adj = adj
        self.device = device
        self.bert_mode = bert_mode
        self.bert_lr = bert_lr
        self.model = make_model(input_dim, bert_mode).to(device)
        self.optimizer = make_optimizer(self.model, bert_lr=bert_lr)
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    # NOTE: in tabular mode GroupDRO re-initialises from scratch every task, so
    # it is a fairness reference point rather than a continual learner and its
    # Acc-BWT reflects single-task training dynamics. In bert_mode it trains
    # continually (see train_task).
    def _reset_model(self):
        self.model = make_model(self.input_dim, self.bert_mode).to(self.device)
        self.optimizer = make_optimizer(self.model, bert_lr=self.bert_lr)

    def train_task(self, X_train: np.ndarray, y_train: np.ndarray,
                   s_train: np.ndarray, epochs: int = MLP_EPOCHS,
                   seen_classes: list[int] | None = None):
        # Fresh per task for tabular; continual for BIOS (class-incremental setup).
        if not self.bert_mode:
            self._reset_model()

        groups = np.unique(s_train)
        n_groups = len(groups)
        group_idx = {g: i for i, g in enumerate(groups)}

        group_counts = np.array([np.sum(s_train == g) for g in groups], dtype=np.float32)
        adj_values = torch.tensor(
            [self.adj / np.sqrt(c) if c > 0 else 0.0 for c in group_counts],
            dtype=torch.float32, device=self.device,
        )

        q = torch.ones(n_groups, device=self.device) / n_groups

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
            for xb, yb, sb in loader:
                xb, yb, sb = xb.to(self.device), yb.to(self.device), sb.to(self.device)

                self.optimizer.zero_grad()
                per_sample_loss = self.criterion(mask_logits(self.model(xb), seen_classes), yb)

                group_losses = torch.zeros(n_groups, device=self.device)
                group_ns = torch.zeros(n_groups, device=self.device)
                for g, gi in group_idx.items():
                    mask = sb == g
                    if mask.sum() > 0:
                        group_losses[gi] = per_sample_loss[mask].mean()
                        group_ns[gi] = mask.sum().float()

                adj_losses = group_losses + adj_values

                with torch.no_grad():
                    q = q * torch.exp(self.eta * adj_losses.detach())
                    q = q / q.sum()

                loss = (q * group_losses).sum()
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

