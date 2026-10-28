"""DER++ with a direct current-batch disparity penalty.

This ablation replaces Fair-DER++'s projection with
``lambda * (L_worst - L_best)`` in the DER++ objective. It uses the same model,
optimizer, replay terms, memory budget, and group-loss surrogate.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import (
    DERPP_ALPHA,
    DERPP_BETA,
    DERPP_FAIR_LAMBDA,
    DERPP_MEM_BATCH,
    DERPP_MEM_SIZE,
    DEVICE,
    MLP_BATCH_SIZE,
    MLP_EPOCHS,
    MLP_GRAD_CLIP,
)
from memory import EpisodicMemory
from model import MLP, batchnorm_eval, has_converged, make_optimizer, make_scheduler, predict


def group_loss_gap(losses, groups, min_group_size=2):
    """Return max-minus-min mean loss over estimable groups."""
    means = [
        losses[groups == group].mean()
        for group in groups.unique()
        if int((groups == group).sum()) >= min_group_size
    ]
    if len(means) < 2:
        return None
    values = torch.stack(means)
    return values.max() - values.min()


class DERppPenalty:
    """DER++ with an additive loss-gap penalty instead of projection geometry."""

    def __init__(
        self,
        input_dim,
        device=DEVICE,
        alpha=DERPP_ALPHA,
        beta=DERPP_BETA,
        fairness_weight=DERPP_FAIR_LAMBDA,
        mem_size=DERPP_MEM_SIZE,
        mem_batch=DERPP_MEM_BATCH,
    ):
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.fairness_weight = fairness_weight
        self.mem_batch_size = int(mem_batch)
        self.model = MLP(input_dim).to(device)
        self.optimizer = make_optimizer(self.model)
        self.memory = EpisodicMemory(size_per_task=int(mem_size), device=device)

    def train_task(self, X_train, y_train, s_train, epochs=MLP_EPOCHS):
        loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long),
                torch.tensor(s_train, dtype=torch.long),
            ),
            batch_size=MLP_BATCH_SIZE,
            shuffle=True,
        )
        scheduler = make_scheduler(self.optimizer, epochs)
        self.model.train()
        epoch_losses = []

        for _ in range(epochs):
            batch_loss = 0.0
            for xb, yb, sb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                sb = sb.to(self.device)
                self.optimizer.zero_grad()

                logits = self.model(xb)
                per_sample = F.cross_entropy(logits, yb, reduction="none")
                loss = per_sample.mean()
                gap = group_loss_gap(per_sample, sb)
                if gap is not None:
                    loss = loss + self.fairness_weight * gap

                mem_distill = self.memory.get_random_batch_with_logits(self.mem_batch_size)
                mem_replay = self.memory.get_random_batch_with_logits(self.mem_batch_size)
                if mem_distill is not None:
                    xd, _yd, _sd, zd = mem_distill
                    xr, yr, _sr, _zr = mem_replay
                    with batchnorm_eval(self.model):
                        logits_d = self.model(xd)
                        logits_r = self.model(xr)
                    loss = (
                        loss
                        + self.alpha * F.mse_loss(logits_d, zd.detach())
                        + self.beta * F.cross_entropy(logits_r, yr)
                    )

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
