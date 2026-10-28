"""
baselines_fairl.py — FaIRL: adversarial fairness via MCR² coding-rate reduction.

Port of FaIRL (Basu Roy Chowdhury & Chaturvedi, AAAI 2023).

Per-step algorithm:
  z = normalize(encoder(x))
  1. Discriminator: L_D = CE(netD(z.detach()), s)   [repeated FAIRL_D_STEP times]
  2. Encoder + classifier:
       mcr_loss = −ΔR(z; Y)                          [maximise coding-rate reduction]
       adv_loss = KL(log_softmax(netD(z)) ‖ uniform) [push z to not predict s]
       cls_loss = CE(classifier(z), y)
       L_G = mcr_loss + λ·adv_loss + cls_loss

Modes:
  tabular (bert_mode=False): MLP encoder, reset fresh each task.
  bios    (bert_mode=True):  BERT→Linear encoder, continual (never reset).
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import (
    MLP_HIDDEN_DIM, MLP_EPOCHS, MLP_LR, MLP_WEIGHT_DECAY,
    MLP_GRAD_CLIP, N_CLASSES, DEVICE,
    FAIRL_LAMBDA, FAIRL_EPS, FAIRL_ADV_DIM, FAIRL_D_STEP, BIOS_BERT_LR,
)
import runctx
from model import make_scheduler, has_converged, predict, mask_logits, _x_tensor, _train_batch_size


def _make_loader(dataset, batch_size, shuffle=True):
    g = torch.Generator(); g.manual_seed(runctx.effective_seed())  # --seed varies batch order
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g)


# MCR² coding-rate helper
class _MCR:
    """Maximal Coding Rate Reduction (Yu et al., 2020), ΔR = R(Z) − R(Z|Y).

    R(Z)   = ½ logdet(I + (d/nε) ZZᵀ)                    [whole-batch volume]
    R(Z|Y) = Σ_k (n_k/2n) logdet(I + (d/n_k ε) Z_k Z_kᵀ) [per-class volume]

    Encoder maximises ΔR (we minimise −ΔR). Features assumed L2-normalised.
    """
    def __init__(self, eps: float = 0.5):
        self.eps = eps

    def _logdet(self, M):
        sign, logabsdet = torch.slogdet(M)
        if sign <= 0:
            return torch.zeros(1, device=M.device, dtype=M.dtype).squeeze()
        return logabsdet

    def compute_discrimn_loss(self, Z):
        # Z: [d, n]
        d, n = Z.shape
        I = torch.eye(d, device=Z.device)
        scalar = d / (n * self.eps)
        return self._logdet(I + scalar * (Z @ Z.t())) / 2.0

    def compute_compress_loss(self, Z, Pi):
        d, n = Z.shape
        I = torch.eye(d, device=Z.device)
        loss = 0.0
        for j in range(Pi.shape[1]):
            Z_j = Z[:, Pi[:, j] == 1]
            n_j = Z_j.shape[1]
            if n_j == 0:
                continue
            scalar = d / (n_j * self.eps)
            loss += (n_j / n) * self._logdet(I + scalar * (Z_j @ Z_j.t())) / 2.0
        return loss

    def deltaR(self, Z, Y, num_classes):
        # Z: [n, d] (L2-normalised rows) → work in [d, n]
        Zt = Z.t()
        Pi = F.one_hot(Y, num_classes).float().to(Z.device)
        R = self.compute_discrimn_loss(Zt)
        Rc = self.compute_compress_loss(Zt, Pi)
        return R - Rc, (R, Rc)


# Encoders
class _MLPEncoder(nn.Module):
    """Tabular encoder: input → H → adv_dim."""
    def __init__(self, input_dim, hidden_dim=MLP_HIDDEN_DIM, adv_dim=FAIRL_ADV_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, adv_dim),
        )

    def forward(self, x):
        return self.net(x)


class _BertEncoder(nn.Module):
    """BERT pooled-CLS output → Linear(768→adv_dim), matching the FaIRL reference bert_model + netG."""
    BERT_DIM = 768

    def __init__(self, adv_dim=FAIRL_ADV_DIM):
        super().__init__()
        from model import build_bert
        self.bert = build_bert()
        if os.environ.get("CLFAIR_NO_GRAD_CKPT") != "1":   # same gate as model.py
            self.bert.gradient_checkpointing_enable()
        self.proj = nn.Linear(self.BERT_DIM, adv_dim)

    def forward(self, x):
        attention_mask = (x != 0).long()
        pooled = self.bert(x, attention_mask=attention_mask)[1]
        return self.proj(pooled)


class _Discriminator(nn.Module):
    """Adversarial sensitive-attribute predictor (netD)."""
    def __init__(self, adv_dim=FAIRL_ADV_DIM, n_sensitive=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(adv_dim, adv_dim), nn.ReLU(), nn.Linear(adv_dim, n_sensitive),
        )

    def forward(self, z):
        return self.net(z)


class _FAIRLNet(nn.Module):
    """
    Wrapper exposing encoder→(normalise)→classifier as one nn.Module, so train.py's
    checkpoint + eval path (.state_dict(), forward(x).argmax) works unchanged.
    Features are L2-normalised before the classifier, matching training.
    """
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def encode(self, x):
        return F.normalize(self.encoder(x), dim=1)

    def forward(self, x):
        return self.classifier(self.encode(x))


# FaIRL
class FaIRL:
    """
    FaIRL-inspired baseline: MCR² task term + adversarial group debiasing.

    Deviations from Basu Roy Chowdhury & Chaturvedi (AAAI 2023):

      * The sensitive-attribute rate-distortion term ΔR(Z; S) — FaIRL's actual
        contribution — is not implemented. Group debiasing here is a
        cross-entropy discriminator plus KL-to-uniform, which is Adversarial
        Removal (Elazar & Goldberg), not FaIRL. MCR² is applied only to the
        class partition. FAIRL_BETA / FAIRL_GAMMA / FAIRL_ETA, the coefficients
        on FaIRL's incremental consistency terms, are unused.
      * `remember()` is a no-op, so this is the only method in the comparison
        with no episodic memory at all, while every other method holds
        MEMORY_SIZE_PER_TASK samples per task. FAIRL_EXEMPLAR_SIZE is unused.
        It therefore has no continual-learning mechanism beyond the objective.
      * `_MCR._logdet` returns a zero gradient on numerical failure without
        warning, so a persistently failing MCR² term would be silently inert.

    Because of these deviations, results are not directly comparable with
    published FaIRL numbers; treat this baseline as an objective-level diagnostic.

    Interface matches the other baselines:
        fairl.train_task(X, y, s, epochs)
        fairl.remember(X, y, s)      # no-op (kept for train.py call parity)
        fairl.predict(X) -> np.ndarray
    Checkpoint: fairl.model.state_dict()/load_state_dict(); fairl._exemplars (unused).
    """

    def __init__(
        self,
        input_dim: int,
        device: str = DEVICE,
        eps: float = FAIRL_EPS,
        lam: float = FAIRL_LAMBDA,
        adv_dim: int = FAIRL_ADV_DIM,
        d_step: int = FAIRL_D_STEP,
        bert_mode: bool = False,
        bert_lr: float = None,
    ):
        self.input_dim = input_dim
        self.device = device
        self.eps = eps
        self.lam = lam
        self.adv_dim = adv_dim
        self.d_step = d_step
        self.bert_mode = bert_mode
        self.bert_lr = bert_lr if bert_lr is not None else BIOS_BERT_LR
        self.n_classes = N_CLASSES
        self._n_sensitive = 2
        self._mcr = _MCR(eps=eps)
        # Kept only so train.py's checkpoint collect/restore has the attribute.
        self._exemplars: list = []

        self._build()

    # Construction / reset
    def _make_encoder(self):
        if self.bert_mode:
            return _BertEncoder(self.adv_dim).to(self.device)
        return _MLPEncoder(self.input_dim, MLP_HIDDEN_DIM, self.adv_dim).to(self.device)

    def _build(self):
        """Create encoder, classifier, discriminator and their optimisers."""
        self.encoder = self._make_encoder()
        self.classifier = nn.Linear(self.adv_dim, self.n_classes).to(self.device)
        self.discriminator = _Discriminator(self.adv_dim, self._n_sensitive).to(self.device)
        self._build_optimisers()
        # Eval/checkpoint wrapper over encoder+classifier
        self.model = _FAIRLNet(self.encoder, self.classifier).to(self.device)

    def _build_optimisers(self):
        if self.bert_mode:
            # Two groups: fine-tune BERT slowly (FaIRL reference bert_lr=2e-5), heads faster.
            self.opt_enc = torch.optim.AdamW([
                {"params": self.encoder.bert.parameters(), "lr": self.bert_lr},
                {"params": self.encoder.proj.parameters(), "lr": MLP_LR},
            ], weight_decay=MLP_WEIGHT_DECAY)
        else:
            self.opt_enc = torch.optim.AdamW(
                self.encoder.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)
        self.opt_cls = torch.optim.AdamW(
            self.classifier.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)
        self.opt_disc = torch.optim.AdamW(
            self.discriminator.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)

    def _reset(self):
        """
        Re-initialise everything for a fresh per-task model (tabular only).

        In bert_mode the model is CONTINUAL — never reset — matching the FaIRL
        reference's sequential bios training (re-initialising pretrained BERT
        would also discard the pretrained weights).
        """
        if self.bert_mode:
            return
        self._build()

    # Training
    def train_task(self, X_train, y_train, s_train, epochs: int = MLP_EPOCHS,
                   seen_classes: list[int] | None = None):
        self._reset()

        X_t = _x_tensor(X_train)
        y_t = torch.tensor(y_train, dtype=torch.long)
        s_t = torch.tensor(s_train, dtype=torch.long)
        loader = _make_loader(
            TensorDataset(X_t, y_t, s_t),
            batch_size=_train_batch_size(self.bert_mode),
        )

        sched_enc = make_scheduler(self.opt_enc, epochs, steps_per_epoch=len(loader))
        self.encoder.train(); self.classifier.train(); self.discriminator.train()
        epoch_losses = []

        for _ in range(epochs):
            batch_loss = 0.0
            for xb, yb, sb in loader:
                xb, yb, sb = xb.to(self.device), yb.to(self.device), sb.to(self.device)

                # Discriminator step(s)
                with torch.no_grad():
                    z_det = F.normalize(self.encoder(xb), dim=1)
                for _ in range(self.d_step):
                    d_loss = F.cross_entropy(self.discriminator(z_det), sb)
                    self.opt_disc.zero_grad()
                    d_loss.backward()
                    nn.utils.clip_grad_norm_(self.discriminator.parameters(), MLP_GRAD_CLIP)
                    self.opt_disc.step()

                # Encoder + classifier step
                z = F.normalize(self.encoder(xb), dim=1)
                mcr_loss = -self._mcr.deltaR(z, yb, self.n_classes)[0]

                d_logp = F.log_softmax(self.discriminator(z), dim=1)
                unif = torch.full_like(d_logp, 1.0 / self._n_sensitive)
                adv_loss = F.kl_div(d_logp, unif, reduction="batchmean")

                cls_loss = F.cross_entropy(mask_logits(self.classifier(z), seen_classes), yb)

                loss = mcr_loss + self.lam * adv_loss + cls_loss
                self.opt_enc.zero_grad(); self.opt_cls.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.encoder.parameters(), MLP_GRAD_CLIP)
                nn.utils.clip_grad_norm_(self.classifier.parameters(), MLP_GRAD_CLIP)
                self.opt_enc.step(); self.opt_cls.step()
                sched_enc.step()
                batch_loss += loss.item()

            epoch_losses.append(batch_loss / len(loader))
            if has_converged(epoch_losses):
                break

        # Refresh eval/checkpoint wrapper to point at the trained modules.
        self.model = _FAIRLNet(self.encoder, self.classifier).to(self.device)
        self.encoder.eval(); self.classifier.eval()

    def remember(self, X, y, s):
        """No-op — MCR² FaIRL keeps no exemplar buffer (kept for call parity)."""
        return

    # Inference
    def predict(self, X, batch_size: int = 256, seen_classes: list[int] | None = None) -> np.ndarray:
        return predict(self.model, X, self.device, batch_size=batch_size,
                       seen_classes=seen_classes)
