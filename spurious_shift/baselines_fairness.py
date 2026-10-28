"""
baselines_fairness.py — Fairness-aware baselines.

  GroupDRO: Distributionally Robust Optimization over demographic groups
            with exponential group weight updates (Sagawa et al., ICLR 2020).
  FAIRL: Fairness-Aware Incremental Representation Learning
         (Chowdhury & Chaturvedi, AAAI 2023). Adversarial debiasing via
         MCR² rate-distortion objective + episodic memory replay for CL.
"""

import copy
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import (MLP, make_optimizer, make_scheduler, has_converged, predict,
                   batchnorm_eval)
from memory import EpisodicMemory
from config import (
    MLP_EPOCHS, MLP_BATCH_SIZE, MLP_LR, MLP_WEIGHT_DECAY,
    MLP_HIDDEN_DIM, MLP_GRAD_CLIP,
    DRO_ETA, DRO_ADJ,
    FAIRL_LAMBDA, FAIRL_EPS, FAIRL_ADV_DIM, FAIRL_MEM_SIZE, FAIRL_MEM_BATCH,
    DEVICE,
)


# MCR^2 utilities for FAIRL

def _reg_logdet(M, ridge=1e-4):
    """log det(M + ridge·I)"""
    n = M.shape[0]
    return torch.logdet(M + ridge * torch.eye(n, dtype=M.dtype, device=M.device))


def _mcr2(Z, group_labels, eps):
    """Calculate mcr^2. Groups with fewer than 2 samples are skipped."""
    n, d = Z.shape
    scale = d / (n * eps ** 2)
    R = 0.5 * _reg_logdet(torch.eye(d, dtype=Z.dtype, device=Z.device) + scale * (Z.T @ Z))
    Rc = torch.zeros(1, dtype=Z.dtype, device=Z.device).squeeze()
    for g in group_labels.unique():
        mask = group_labels == g
        n_g = int(mask.sum())
        if n_g < 2:
            continue
        Z_g = Z[mask]
        scale_g = d / (n_g * eps ** 2)
        Rc = Rc + (n_g / (2.0 * n)) * _reg_logdet(
            torch.eye(d, dtype=Z.dtype, device=Z.device) + scale_g * (Z_g.T @ Z_g)
        )
    return R - Rc


def _forgetting_loss(Z_cur, Z_frz, y_mem, eps):
    """
    Equation 5 from the FAIRL paper: measures encoder representation drift,
    partitioned by target class.
    """
    total = torch.zeros(1, dtype=Z_cur.dtype, device=Z_cur.device).squeeze()
    for c in y_mem.unique():
        mask = y_mem == c
        Z_c = Z_cur[mask]
        Z_bar_c = Z_frz[mask]
        if Z_c.shape[0] < 2 or Z_bar_c.shape[0] < 2:
            continue
        d, n_c, n_bar_c  = Z_c.shape[1], Z_c.shape[0], Z_bar_c.shape[0]
        Z_joint = torch.cat([Z_c, Z_bar_c], dim=0)
        n_joint = n_c + n_bar_c

        def _R(Z_set, n_set):
            scale = d / (n_set * eps ** 2)
            return 0.5 * _reg_logdet(
                torch.eye(d, dtype=Z_set.dtype, device=Z_set.device) + scale * (Z_set.T @ Z_set)
            )
        total = total + (_R(Z_joint, n_joint) - 0.5 * (_R(Z_c, n_c) + _R(Z_bar_c, n_bar_c)))
    return total


class GroupDRO:
    """
    Group DRO implementation

    Uses exponential group weight updates with a generalization adjustment
    (adds C/sqrt(n_g) per group to account for smaller groups being harder to generalize to)
    """
    def __init__(self, input_dim, eta=DRO_ETA, adj=DRO_ADJ, device=DEVICE):
        self.input_dim = input_dim
        self.eta = eta
        self.adj = adj
        self.device = device
        self.model = MLP(input_dim).to(device)
        self.optimizer = make_optimizer(self.model)
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    # NOTE: GroupDRO re-initialises from scratch every task — it is a fairness
    # reference point, not a continual learner. Its Acc-BWT in the tables
    # measures single-task training dynamics, not forgetting.
    def _reset_model(self):
        self.model = MLP(self.input_dim).to(self.device)
        self.optimizer = make_optimizer(self.model)

    def train_task(self, X_train, y_train, s_train, epochs=MLP_EPOCHS):
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
            for xb, yb, sb in loader:
                xb, yb, sb = xb.to(self.device), yb.to(self.device), sb.to(self.device)
                self.optimizer.zero_grad()
                per_sample_loss = self.criterion(self.model(xb), yb)

                group_losses = torch.zeros(n_groups, device=self.device)
                for g, gi in group_idx.items():
                    mask = sb == g
                    if mask.sum() > 0:
                        group_losses[gi] = per_sample_loss[mask].mean()

                with torch.no_grad():
                    q = q * torch.exp(self.eta * (group_losses + adj_values).detach())
                    q = q / q.sum()

                loss = (q * group_losses).sum()
                loss.backward()
                # Clip to MLP_GRAD_CLIP like every other method here; the
                # exponentiated group weights can make gradients spike.
                nn.utils.clip_grad_norm_(self.model.parameters(), MLP_GRAD_CLIP)
                self.optimizer.step()
                batch_loss += loss.item()
            scheduler.step()
            epoch_losses.append(batch_loss / len(loader))
            if has_converged(epoch_losses):
                break

    def predict(self, X):
        return predict(self.model, X, self.device)



class FAIRL:
    """
    FaIRL-inspired baseline (Roy Chowdhury & Chaturvedi, AAAI 2023).

    Deviations from the published method:

      * The model is initialised lazily on task 0 and weights carry forward
        on tasks 1+ (`_reset_model()` early-returns once the optimizer
        exists). Retention comes from the carried weights, the replay term,
        and the frozen-encoder forgetting loss. (GroupDRO, by contrast,
        re-initialises every task, so its BWT measures single-task training,
        not forgetting.)
      * One discriminator step per generator step, against 5 in the reference
        implementation. An under-trained adversary weakens the debiasing signal.
      * MCR^2 is computed on unnormalised penultimate features. The objective is
        only well-posed on L2-normalised features; unnormalised, it is unbounded
        below and rewards inflating the feature norm.

    Because of these deviations the results are not a faithful comparison with
    published FaIRL and serve only as an objective-level diagnostic.
    """
    def __init__(self, input_dim, device=DEVICE, lam=FAIRL_LAMBDA, eps=FAIRL_EPS,
                 adv_dim=FAIRL_ADV_DIM, mem_size=FAIRL_MEM_SIZE, mem_batch=FAIRL_MEM_BATCH):
        self.input_dim = input_dim
        self.device = device
        self.lam = lam
        self.eps = eps
        self.adv_dim = adv_dim
        self.mem_batch_size = int(mem_batch)
        self.model = MLP(input_dim).to(device)
        self.adversary = None
        self.optimizer = None
        self.adv_optimizer = None
        self._frozen_model = None
        self.memory = EpisodicMemory(size_per_task=int(mem_size), device=device)

    def _reset_model(self):
        if self.optimizer is not None:
            return  # carry forward weights on tasks 1+
        self.model = MLP(self.input_dim).to(self.device)
        self.adversary = nn.Linear(MLP_HIDDEN_DIM, self.adv_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)
        self.adv_optimizer = torch.optim.AdamW(self.adversary.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)

    def _penultimate(self, x):
        # net[:8] = output of the second Dropout; net[8] = final Linear
        return self.model.net[:8](x)

    def train_task(self, X_train, y_train, s_train, epochs=MLP_EPOCHS):
        self._reset_model()

        has_memory = self.memory.get_random_batch(1) is not None
        if has_memory:
            self._frozen_model = copy.deepcopy(self.model)
            self._frozen_model.eval()
            for p in self._frozen_model.parameters():
                p.requires_grad_(False)

        loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.long),
                          torch.tensor(s_train, dtype=torch.long)),
            batch_size=MLP_BATCH_SIZE, shuffle=True,
        )
        scheduler = make_scheduler(self.optimizer, epochs)
        self.model.train()
        self.adversary.train()
        epoch_losses = []

        for _ in range(epochs):
            enc_epoch_loss = 0.0
            for xb, yb, sb in loader:
                xb, yb, sb = xb.to(self.device), yb.to(self.device), sb.to(self.device)

                # D step:
                self.adv_optimizer.zero_grad()
                with torch.no_grad():
                    h_det = self._penultimate(xb)
                adv_loss = -_mcr2(self.adversary(h_det), sb, self.eps)
                adv_loss.backward()
                nn.utils.clip_grad_norm_(self.adversary.parameters(), MLP_GRAD_CLIP)
                self.adv_optimizer.step()

                # phi step:
                self.optimizer.zero_grad()
                h = self._penultimate(xb)
                z_prime = self.adversary(h)
                logits = self.model.net[8](h)

                enc_loss = (
                    nn.CrossEntropyLoss()(logits, yb)
                    - _mcr2(h, yb, self.eps)                   # (a) task discriminability
                    + self.lam * _mcr2(z_prime, sb, self.eps)  # (b) minimize group discriminability
                )

                mem = self.memory.get_random_batch(self.mem_batch_size)
                if mem is not None:
                    xm, ym, sm = mem
                    # Replay forward with BatchNorm in eval mode, matching A-GEM
                    # and DER++, so the replay batch does not update the
                    # running statistics.
                    with batchnorm_eval(self.model):
                        hm = self._penultimate(xm)
                        z_prime_m = self.adversary(hm)
                        logits_m = self.model.net[8](hm)
                        enc_loss = enc_loss + nn.CrossEntropyLoss()(logits_m, ym)
                        enc_loss = enc_loss + self.lam * _mcr2(z_prime_m, sm, self.eps)
                        with torch.no_grad():
                            h_frz = self._frozen_model.net[:8](xm)
                        enc_loss = enc_loss + _forgetting_loss(hm, h_frz, ym, self.eps)

                enc_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), MLP_GRAD_CLIP)
                self.optimizer.step()
                enc_epoch_loss += enc_loss.item()

            scheduler.step()
            epoch_losses.append(enc_epoch_loss / len(loader))
            if has_converged(epoch_losses):
                break

    def remember(self, X, y, s):
        self.memory.add_task(X, y, s)

    def predict(self, X):
        return predict(self.model, X, self.device)
