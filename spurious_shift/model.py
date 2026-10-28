"""model.py — Shared MLP backbone used by all neural baselines and FAIR-GEM."""

import os
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MLP_HIDDEN_DIM, MLP_DROPOUT, MLP_LR, MLP_WEIGHT_DECAY, \
                   MLP_WARMUP_EPOCHS, N_CLASSES, MLP_PATIENCE, MLP_MIN_DELTA


# How an auxiliary forward pass treats BatchNorm's running statistics.
#   freeze  normalise by batch statistics, but do not update the running
#           estimates (momentum=0). The default and the published protocol.
#   eval    normalise by the running estimates carried over from earlier tasks.
#           A different method, not a different spelling — see norm_study.csv.
#   legacy  auxiliary passes update the running estimates. Diagnostic only.
BN_MODES = ("freeze", "eval", "legacy")


def _norm_layer(width):
    """
    Normalisation used by the MLP. CLFAIR_NORM selects it.

    Used to test whether the disparity safety gradient depends on BatchNorm
    specifically. BatchNorm normalises using statistics pooled over the batch,
    so gradients couple samples — including across demographic groups — and the
    safety gradient can act through that coupling. LayerNorm normalises per
    sample, so no such channel exists, and neither does any running state. This
    matters because the Bias-in-Bios encoder (BERT) uses LayerNorm throughout.
    """
    kind = os.environ.get("CLFAIR_NORM", "batch")
    if kind == "batch":
        return nn.BatchNorm1d(width)
    if kind == "layer":
        return nn.LayerNorm(width)
    if kind == "none":
        return nn.Identity()
    raise SystemExit(f"CLFAIR_NORM must be batch|layer|none, got {kind!r}")


class MLP(nn.Module):
    """3-layer MLP: input -> H -> H -> N_CLASSES, with normalisation and Dropout."""

    def __init__(self, input_dim, hidden_dim=MLP_HIDDEN_DIM, output_dim=N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _norm_layer(hidden_dim),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
            nn.Linear(hidden_dim, hidden_dim),
            _norm_layer(hidden_dim),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


    def get_flat_grad(self):
        grads = []
        for p in self.parameters():
            if p.grad is None:
                grads.append(torch.zeros_like(p.data).view(-1))
            else:
                grads.append(p.grad.view(-1))
        return torch.cat(grads)

    def set_flat_grad(self, flat_grad):
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.grad = flat_grad[offset: offset + numel].view(p.shape)
            offset += numel


@contextmanager
def batchnorm_eval(model):
    """
    Run an auxiliary forward pass with BatchNorm in eval mode.

    Only the *main* task forward should update BatchNorm's running statistics.
    Auxiliary passes — the replay/memory gradient and the disparity safety
    gradient — must not, or a method that computes more of them accumulates more
    momentum updates per step than one that computes fewer.

    Restores each module's prior mode, so nesting and use outside training are
    both safe. CLFAIR_BN_MODE selects between BN_MODES; see that constant.
    """
    # Default "freeze", chosen by the normalisation ablation (norm_study.csv):
    # keep batch-statistic normalisation so the auxiliary gradient measures
    # disparity on the current batch, while momentum=0 stops the running
    # estimates being updated. "eval" — normalising the auxiliary pass by
    # running statistics carried over from earlier tasks — was worst in every
    # condition, including after retuning.
    #
    # An unrecognised CLFAIR_BN_MODE raises rather than falling back to a
    # default, since the modes are different methods.
    mode = os.environ.get("CLFAIR_BN_MODE", "freeze")
    if mode not in BN_MODES:
        raise SystemExit(
            f"CLFAIR_BN_MODE must be {'|'.join(sorted(BN_MODES))}, got {mode!r}")
    if mode == "freeze":
        # Keep train-mode normalisation — the auxiliary forward still
        # normalises by batch statistics and gradients still flow through them —
        # while momentum=0 makes the running-statistics update a no-op:
        #     running = (1 - 0) * running + 0 * batch   ->  unchanged
        # This separates the two effects that switching to eval() bundles: the
        # statistics update and the change in what the gradient computes.
        #
        # Restoring the buffers by hand instead would corrupt autograd: the
        # context exits before backward(), so an in-place write bumps the tensor
        # version and the graph is invalidated. momentum=0 avoids any extra write.
        bns = [m for m in model.modules()
               if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        prior_m = [m.momentum for m in bns]
        prior_n = [m.num_batches_tracked.item() if m.num_batches_tracked is not None
                   else None for m in bns]
        for m in bns:
            m.momentum = 0.0
        try:
            yield
        finally:
            for m, mom, nb in zip(bns, prior_m, prior_n):
                m.momentum = mom
                # num_batches_tracked is an integer buffer, not part of the float
                # backward graph, so restoring it via .data is safe and keeps the
                # "no update happened" invariant checkable.
                if nb is not None:
                    m.num_batches_tracked.data.fill_(nb)
        return
    if mode == "legacy":
        # Diagnostic only: auxiliary passes update BatchNorm's running
        # statistics.
        yield
        return
    bns = [m for m in model.modules()
           if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    prior = [m.training for m in bns]
    for m in bns:
        m.eval()
    try:
        yield
    finally:
        for m, was_training in zip(bns, prior):
            m.train(was_training)


def make_optimizer(model, lr=None, weight_decay=None):
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr if lr is not None else MLP_LR,
        weight_decay=weight_decay if weight_decay is not None else MLP_WEIGHT_DECAY,
    )


def make_scheduler(optimizer, n_epochs):
    """Linear warmup for MLP_WARMUP_EPOCHS, then cosine annealing to 1% of peak LR."""
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=MLP_WARMUP_EPOCHS
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, n_epochs - MLP_WARMUP_EPOCHS), eta_min=MLP_LR * 0.01
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[MLP_WARMUP_EPOCHS]
    )


def has_converged(loss_history, patience=MLP_PATIENCE, min_delta=MLP_MIN_DELTA):
    """True if the last 'patience' losses have all stayed within 'min_delta' of each other."""
    if len(loss_history) < patience + 1:
        return False
    recent = loss_history[-(patience + 1):]
    return (max(recent) - min(recent)) < min_delta


def predict(model, X, device="cpu"):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(device))
        return logits.argmax(dim=1).cpu().numpy()




def per_sample_ce(model, X, y, device="cpu", batch_size=1024):
    """Per-sample cross-entropy, eval mode, no grad (surrogate-validation path)."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    out = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            out.append(F.cross_entropy(model(X_t[i:i + batch_size].to(device)),
                                       y_t[i:i + batch_size].to(device),
                                       reduction="none").cpu())
    return torch.cat(out).numpy()
