"""
model.py — Shared MLP / BertMLP backbone used by all neural methods.

For bios (bert_mode=True): use BertMLP instead of MLP.
  - BERT encoder runs at train time (not pre-cached).
  - Two optimizer param groups: BERT (BIOS_BERT_LR) + head (MLP_LR).
  - Flat-grad ops operate on all parameters so projection covers the full model.
"""
import os
import sys
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

# CLFAIR_TF32=1 enables TF32 matmuls (Ampere+). ~1.2-1.5x on BERT; results
# differ from FP32 within seed noise, so use it for all runs of an experiment
# or for none.
if os.environ.get("CLFAIR_TF32") == "1":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
from config import MLP_HIDDEN_DIM, MLP_DROPOUT, MLP_LR, MLP_WEIGHT_DECAY, MLP_WARMUP_EPOCHS, N_CLASSES, MLP_PATIENCE, MLP_MIN_DELTA, BIOS_BERT_LR, MLP_BATCH_SIZE, BIOS_BERT_TRAIN_BATCH


@contextmanager
def batchnorm_eval(model):
    """
    Run an auxiliary forward pass with normalisation layers in eval mode.

    Only the *main* task forward should update running statistics. Auxiliary
    passes — the replay/memory gradient and the disparity safety gradient —
    must not, or a method that computes more of them accumulates more momentum
    updates per step than one that computes fewer, and the methods stop being
    comparable at inference time.

    Covers BatchNorm only. BERT normalises with LayerNorm, which holds no
    running statistics and is unaffected by train/eval mode, so the classifier
    head is the part that matters here.

    Restores each module's prior mode, so nesting and use outside training are
    both safe.
    """
    # Freeze the running estimates without changing the normalisation the
    # forward pass uses: momentum=0 makes the update
    #     running = (1 - 0) * running + 0 * batch
    # a no-op, while activations are still normalised by batch statistics.
    #
    # Switching to eval() here instead would also change what the auxiliary
    # gradient computes — it would normalise by running statistics estimated on
    # earlier tasks, so the measured group disparity would not correspond to the
    # batch being examined. In the Spurious Shift normalisation ablation this
    # roughly tripled the accuracy-parity gap, also after retuning.
    mode = os.environ.get("CLFAIR_BN_MODE", "freeze")
    bns = [m for m in model.modules()
           if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if mode == "legacy":
        yield
        return
    if mode == "eval":
        prior = [m.training for m in bns]
        for m in bns:
            m.eval()
        try:
            yield
        finally:
            for m, was_training in zip(bns, prior):
                m.train(was_training)
        return
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
            if nb is not None:
                m.num_batches_tracked.data.fill_(nb)


class MLP(nn.Module):
    """3-layer MLP: input → H → H → N_CLASSES with BatchNorm + Dropout."""

    def __init__(self, input_dim: int, hidden_dim: int = MLP_HIDDEN_DIM, output_dim: int = N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


    def get_flat_grad(self) -> torch.Tensor:
        grads = []
        for p in self.parameters():
            if p.grad is None:
                grads.append(torch.zeros_like(p.data).view(-1))
            else:
                grads.append(p.grad.view(-1))
        return torch.cat(grads)

    def set_flat_grad(self, flat_grad: torch.Tensor):
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.grad = flat_grad[offset: offset + numel].view(p.shape)
            offset += numel



def build_bert():
    """Construct the BERT encoder, optionally skipping the pretrained download.

    Eval scripts restore a checkpoint immediately after construction, and
    load_state_dict is strict, so every pretrained tensor is overwritten before
    it is ever used. Fetching them first is pure overhead -- six eval stages x
    nine methods is 54 redundant 440 MB loads per seed. CLFAIR_SKIP_PRETRAINED=1
    builds from the config alone instead; the restored weights are identical.

    Never set this for training, where the pretrained initialisation is the
    whole point. train.py refuses to start with it set.
    """
    import io, transformers
    from transformers import BertModel as _BertModel, BertConfig as _BertConfig
    _prev = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    _old_stdout, sys.stdout = sys.stdout, io.StringIO()
    try:
        if os.environ.get("CLFAIR_SKIP_PRETRAINED") == "1":
            return _BertModel(_BertConfig.from_pretrained("bert-base-uncased"))
        return _BertModel.from_pretrained("bert-base-uncased")
    finally:
        sys.stdout = _old_stdout
        transformers.logging.set_verbosity(_prev)


def mask_logits(logits: torch.Tensor, seen_classes: list[int] | None) -> torch.Tensor:
    """Zero out unseen-class logits for class-incremental learning.

    Restricts softmax to seen classes during training so new-task CE doesn't
    suppress old-class logits. Pass None for domain-IL (spurious_shift).
    """
    if seen_classes is None:
        return logits
    mask = torch.full(logits.shape, float('-inf'), device=logits.device)
    mask[:, seen_classes] = 0.0
    return logits + mask


def _train_batch_size(bert_mode: bool = False) -> int:
    return BIOS_BERT_TRAIN_BATCH if bert_mode else MLP_BATCH_SIZE


def _x_tensor(X: np.ndarray) -> torch.Tensor:
    """Convert numpy X to tensor. Token-ID arrays (int32/int64) → long; else float32."""
    if X.dtype in (np.int32, np.int64):
        return torch.tensor(X, dtype=torch.long)
    return torch.tensor(X, dtype=torch.float32)


class BertMLP(nn.Module):
    """BERT encoder + MLP classification head for BIOS bert_mode.

    Architecture: BertModel(pooler_output)[768] → Linear→BN→ReLU→Dropout (×2) → N_CLASSES.
    Interface is identical to MLP so all method classes work unchanged.
    Flat-grad ops cover all parameters (~110M total) so gradient projection
    and Fisher estimation apply to the full model.
    """
    BERT_DIM = 768

    def __init__(self, hidden_dim: int = MLP_HIDDEN_DIM,
                 output_dim: int = N_CLASSES, dropout: float = MLP_DROPOUT):
        super().__init__()
        self.bert = build_bert()
        # Gradient checkpointing reduces activation memory ~40% at a small compute
        # cost. At seq-32/batch-256 on a >=24GB card the memory saving is
        # unnecessary, so CLFAIR_NO_GRAD_CKPT=1 skips it for a ~1.3x step
        # speedup — numerically identical either way (pure recompute).
        if os.environ.get("CLFAIR_NO_GRAD_CKPT") != "1":
            self.bert.gradient_checkpointing_enable()
        # CLFAIR_NORM=layer swaps the head's BatchNorm1d for LayerNorm (same
        # switch name as the Spurious Shift suite). BatchNorm running stats are
        # only ever updated by the current task's batches, so at eval time the
        # deployed head normalises every task by the LAST task's statistics —
        # a recency artifact. LayerNorm is stateless and removes it.
        norm_kind = os.environ.get("CLFAIR_NORM", "batch")
        _norms = {"batch": nn.BatchNorm1d, "layer": nn.LayerNorm, "none": nn.Identity}
        if norm_kind not in _norms:
            raise ValueError(f"CLFAIR_NORM={norm_kind!r}; expected batch|layer|none")
        _norm = _norms[norm_kind]
        self.head = nn.Sequential(
            nn.Linear(self.BERT_DIM, hidden_dim),
            _norm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            _norm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return BERT pooled-CLS output. Derives attention mask from non-pad tokens."""
        attention_mask = (x != 0).long()
        return self.bert(x, attention_mask=attention_mask)[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))

    def _all_params(self):
        return list(self.parameters())


    def get_flat_grad(self) -> torch.Tensor:
        grads = []
        for p in self._all_params():
            if p.grad is None:
                grads.append(torch.zeros_like(p.data).view(-1))
            else:
                grads.append(p.grad.view(-1))
        return torch.cat(grads)

    def set_flat_grad(self, flat_grad: torch.Tensor):
        offset = 0
        for p in self._all_params():
            numel = p.numel()
            p.grad = flat_grad[offset: offset + numel].view(p.shape)
            offset += numel



def make_model(input_dim: int, bert_mode: bool = False) -> nn.Module:
    """Construct MLP or BertMLP depending on the dataset mode."""
    if bert_mode:
        return BertMLP()
    return MLP(input_dim)


def make_optimizer(model: nn.Module, lr: float = None, weight_decay: float = None,
                   bert_lr: float = None):
    """AdamW optimizer. BertMLP gets two param groups (BERT lr vs. head lr)."""
    wd = weight_decay if weight_decay is not None else MLP_WEIGHT_DECAY
    if isinstance(model, BertMLP):
        b_lr = bert_lr if bert_lr is not None else BIOS_BERT_LR
        return torch.optim.AdamW([
            {"params": model.bert.parameters(), "lr": b_lr, "weight_decay": wd},
            {"params": model.head.parameters(), "lr": lr if lr is not None else MLP_LR, "weight_decay": wd},
        ])
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr if lr is not None else MLP_LR,
        weight_decay=wd,
    )


def make_scheduler(optimizer, n_epochs: int, steps_per_epoch: int | None = None,
                   warmup_frac: float = 0.06):
    """Linear warmup then cosine annealing to 0.

    If steps_per_epoch is given, stepping is per batch (recommended for BERT);
    warmup spans warmup_frac of total steps so it always completes regardless
    of epoch count. Otherwise, stepping is per epoch (MLP mode).
    """
    if steps_per_epoch is not None:
        total_steps = max(1, int(n_epochs) * int(steps_per_epoch))
        if total_steps <= 1:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=0
            )
        warmup_steps = min(max(1, round(warmup_frac * total_steps)), total_steps - 1)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=0
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=MLP_WARMUP_EPOCHS
    )
    cosine_epochs = max(1, n_epochs - MLP_WARMUP_EPOCHS)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=0
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[MLP_WARMUP_EPOCHS]
    )


def has_converged(
    loss_history: list,
    patience: int = MLP_PATIENCE,
    min_delta: float = MLP_MIN_DELTA,
) -> bool:
    """True if the last `patience` losses all stayed within `min_delta` of each other."""
    if len(loss_history) < patience + 1:
        return False
    recent = loss_history[-(patience + 1):]
    return (max(recent) - min(recent)) < min_delta


def predict(model: nn.Module, X: "np.ndarray", device: str = "cpu",
            batch_size: int = 256, seen_classes: list[int] | None = None) -> "np.ndarray":
    model.eval()
    X_t = _x_tensor(X)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            batch = X_t[i: i + batch_size].to(device)
            logits = model(batch)
            if seen_classes is not None:
                mask = torch.full(logits.shape, float('-inf'), device=logits.device)
                mask[:, seen_classes] = 0.0
                logits = logits + mask
            preds.append(logits.argmax(dim=1).cpu())
    return torch.cat(preds).numpy()




def per_sample_ce(model: nn.Module, X: "np.ndarray", y: "np.ndarray", device: str = "cpu",
                  batch_size: int | None = None, seen_classes: list[int] | None = None) -> "np.ndarray":
    """Per-sample cross-entropy on (masked) logits, eval mode, no grad.

    Feeds the surrogate-validation metrics; predict() discards logits, so this
    is the loss path for evaluation.
    """
    batch_size = batch_size or int(os.environ.get("CLFAIR_EVAL_BATCH", "256"))
    model.eval()
    X_t = _x_tensor(X)
    y_t = torch.tensor(y, dtype=torch.long)
    out = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            logits = mask_logits(model(X_t[i:i + batch_size].to(device)), seen_classes)
            out.append(F.cross_entropy(logits, y_t[i:i + batch_size].to(device),
                                       reduction="none").cpu())
    return torch.cat(out).numpy()


def predict_and_ce(model: nn.Module, X: "np.ndarray", y: "np.ndarray", device: str = "cpu",
                   batch_size: int | None = None, seen_classes: list[int] | None = None):
    """One eval forward -> (argmax predictions, per-sample CE).

    Outputs are identical to predict() followed by per_sample_ce(); fusing them
    halves evaluation forwards, which matters when the encoder is BERT.
    """
    batch_size = batch_size or int(os.environ.get("CLFAIR_EVAL_BATCH", "1024"))
    model.eval()
    X_t = _x_tensor(X)
    y_t = torch.tensor(y, dtype=torch.long)
    preds, losses = [], []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            logits = mask_logits(model(X_t[i:i + batch_size].to(device)), seen_classes)
            preds.append(logits.argmax(dim=1).cpu())
            losses.append(F.cross_entropy(logits, y_t[i:i + batch_size].to(device),
                                          reduction="none").cpu())
    return torch.cat(preds).numpy(), torch.cat(losses).numpy()
